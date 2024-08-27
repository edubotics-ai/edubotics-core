# code modified from https://github.com/langchain-ai/langchain/blob/master/cookbook/RAPTOR.ipynb

from typing import Dict, List, Optional, Tuple
import os
import numpy as np
import pandas as pd
import umap
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sklearn.mixture import GaussianMixture
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from edubotics_core.vectorstore.base import VectorStoreBase

RANDOM_SEED = 42


class FAISS(FAISS):
    """To add length property to FAISS class"""

    def __len__(self):
        return self.index.ntotal


class RAPTORVectoreStore(VectorStoreBase):
    def __init__(self, config, documents=[], text_splitter=None, embedding_model=None):
        self.documents = documents
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=self.config["splitter_options"]["chunk_size"],
            chunk_overlap=self.config["splitter_options"]["chunk_overlap"],
            separators=self.config["splitter_options"]["chunk_separators"],
            disallowed_special=(),
        )
        self.embd = embedding_model
        self.model = ChatOpenAI(
            model="gpt-3.5-turbo",
        )

    def concat_documents(self, documents):
        d_sorted = sorted(documents, key=lambda x: x.metadata["source"])
        d_reversed = list(reversed(d_sorted))
        concatenated_content = "\n\n\n --- \n\n\n".join(
            [doc.page_content for doc in d_reversed]
        )
        return concatenated_content

    def split_documents(self, documents):
        concatenated_content = self.concat_documents(documents)
        texts_split = self.text_splitter.split_text(concatenated_content)
        return texts_split

    def add_documents(self, documents):
        self.documents.extend(documents)

    def global_cluster_embeddings(
        self,
        embeddings: np.ndarray,
        dim: int,
        n_neighbors: Optional[int] = None,
        metric: str = "cosine",
    ) -> np.ndarray:
        """
        Perform global dimensionality reduction on the embeddings using UMAP.

        Parameters:
        - embeddings: The input embeddings as a numpy array.
        - dim: The target dimensionality for the reduced space.
        - n_neighbors: Optional; the number of neighbors to consider for each point.
                    If not provided, it defaults to the square root of the number of embeddings.
        - metric: The distance metric to use for UMAP.

        Returns:
        - A numpy array of the embeddings reduced to the specified dimensionality.
        """
        if n_neighbors is None:
            n_neighbors = int((len(embeddings) - 1) ** 0.5)
        return umap.UMAP(
            n_neighbors=n_neighbors, n_components=dim, metric=metric
        ).fit_transform(embeddings)

    def local_cluster_embeddings(
        self,
        embeddings: np.ndarray,
        dim: int,
        num_neighbors: int = 10,
        metric: str = "cosine",
    ) -> np.ndarray:
        """
        Perform local dimensionality reduction on the embeddings using UMAP, typically after global clustering.

        Parameters:
        - embeddings: The input embeddings as a numpy array.
        - dim: The target dimensionality for the reduced space.
        - num_neighbors: The number of neighbors to consider for each point.
        - metric: The distance metric to use for UMAP.

        Returns:
        - A numpy array of the embeddings reduced to the specified dimensionality.
        """
        return umap.UMAP(
            n_neighbors=num_neighbors, n_components=dim, metric=metric
        ).fit_transform(embeddings)

    def get_optimal_clusters(
        self,
        embeddings: np.ndarray,
        max_clusters: int = 50,
        random_state: int = RANDOM_SEED,
    ) -> int:
        """
        Determine the optimal number of clusters using the Bayesian Information Criterion (BIC) with a Gaussian Mixture Model.

        Parameters:
        - embeddings: The input embeddings as a numpy array.
        - max_clusters: The maximum number of clusters to consider.
        - random_state: Seed for reproducibility.

        Returns:
        - An integer representing the optimal number of clusters found.
        """
        max_clusters = min(max_clusters, len(embeddings))
        n_clusters = np.arange(1, max_clusters)
        bics = []
        for n in n_clusters:
            gm = GaussianMixture(n_components=n, random_state=random_state)
            gm.fit(embeddings)
            bics.append(gm.bic(embeddings))
        return n_clusters[np.argmin(bics)]

    def GMM_cluster(
        self, embeddings: np.ndarray, threshold: float, random_state: int = 0
    ):
        """
        Cluster embeddings using a Gaussian Mixture Model (GMM) based on a probability threshold.

        Parameters:
        - embeddings: The input embeddings as a numpy array.
        - threshold: The probability threshold for assigning an embedding to a cluster.
        - random_state: Seed for reproducibility.

        Returns:
        - A tuple containing the cluster labels and the number of clusters determined.
        """
        n_clusters = self.get_optimal_clusters(embeddings)
        gm = GaussianMixture(n_components=n_clusters, random_state=random_state)
        gm.fit(embeddings)
        probs = gm.predict_proba(embeddings)
        labels = [np.where(prob > threshold)[0] for prob in probs]
        return labels, n_clusters

    def perform_clustering(
        self,
        embeddings: np.ndarray,
        dim: int,
        threshold: float,
    ) -> List[np.ndarray]:
        """
        Perform clustering on the embeddings by first reducing their dimensionality globally, then clustering
        using a Gaussian Mixture Model, and finally performing local clustering within each global cluster.

        Parameters:
        - embeddings: The input embeddings as a numpy array.
        - dim: The target dimensionality for UMAP reduction.
        - threshold: The probability threshold for assigning an embedding to a cluster in GMM.

        Returns:
        - A list of numpy arrays, where each array contains the cluster IDs for each embedding.
        """
        if len(embeddings) <= dim + 1:
            # Avoid clustering when there's insufficient data
            return [np.array([0]) for _ in range(len(embeddings))]

        # Global dimensionality reduction
        reduced_embeddings_global = self.global_cluster_embeddings(embeddings, dim)
        # Global clustering
        global_clusters, n_global_clusters = self.GMM_cluster(
            reduced_embeddings_global, threshold
        )

        all_local_clusters = [np.array([]) for _ in range(len(embeddings))]
        total_clusters = 0

        # Iterate through each global cluster to perform local clustering
        for i in range(n_global_clusters):
            # Extract embeddings belonging to the current global cluster
            global_cluster_embeddings_ = embeddings[
                np.array([i in gc for gc in global_clusters])
            ]

            if len(global_cluster_embeddings_) == 0:
                continue
            if len(global_cluster_embeddings_) <= dim + 1:
                # Handle small clusters with direct assignment
                local_clusters = [np.array([0]) for _ in global_cluster_embeddings_]
                n_local_clusters = 1
            else:
                # Local dimensionality reduction and clustering
                reduced_embeddings_local = self.local_cluster_embeddings(
                    global_cluster_embeddings_, dim
                )
                local_clusters, n_local_clusters = self.GMM_cluster(
                    reduced_embeddings_local, threshold
                )

            # Assign local cluster IDs, adjusting for total clusters already processed
            for j in range(n_local_clusters):
                local_cluster_embeddings_ = global_cluster_embeddings_[
                    np.array([j in lc for lc in local_clusters])
                ]
                indices = np.where(
                    (embeddings == local_cluster_embeddings_[:, None]).all(-1)
                )[1]
                for idx in indices:
                    all_local_clusters[idx] = np.append(
                        all_local_clusters[idx], j + total_clusters
                    )

            total_clusters += n_local_clusters

        return all_local_clusters

    def embed(self, texts):
        """
        Generate embeddings for a list of text documents.

        This function assumes the existence of an `embd` object with a method `embed_documents`
        that takes a list of texts and returns their embeddings.

        Parameters:
        - texts: List[str], a list of text documents to be embedded.

        Returns:
        - numpy.ndarray: An array of embeddings for the given text documents.
        """
        text_embeddings = self.embd.embed_documents(texts)
        text_embeddings_np = np.array(text_embeddings)
        return text_embeddings_np

    def embed_cluster_texts(self, texts):
        """
        Embeds a list of texts and clusters them, returning a DataFrame with texts, their embeddings, and cluster labels.

        This function combines embedding generation and clustering into a single step. It assumes the existence
        of a previously defined `perform_clustering` function that performs clustering on the embeddings.

        Parameters:
        - texts: List[str], a list of text documents to be processed.

        Returns:
        - pandas.DataFrame: A DataFrame containing the original texts, their embeddings, and the assigned cluster labels.
        """
        text_embeddings_np = self.embed(texts)  # Generate embeddings
        cluster_labels = self.perform_clustering(
            text_embeddings_np, 10, 0.1
        )  # Perform clustering on the embeddings
        df = pd.DataFrame()  # Initialize a DataFrame to store the results
        df["text"] = texts  # Store original texts
        df["embd"] = list(
            text_embeddings_np
        )  # Store embeddings as a list in the DataFrame
        df["cluster"] = cluster_labels  # Store cluster labels
        return df

    def fmt_txt(self, df: pd.DataFrame) -> str:
        """
        Formats the text documents in a DataFrame into a single string.

        Parameters:
        - df: DataFrame containing the 'text' column with text documents to format.

        Returns:
        - A single string where all text documents are joined by a specific delimiter.
        """
        unique_txt = df["text"].tolist()
        return "--- --- \n --- --- ".join(unique_txt)

    def embed_cluster_summarize_texts(
        self, texts: List[str], level: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Embeds, clusters, and summarizes a list of texts. This function first generates embeddings for the texts,
        clusters them based on similarity, expands the cluster assignments for easier processing, and then summarizes
        the content within each cluster.

        Parameters:
        - texts: A list of text documents to be processed.
        - level: An integer parameter that could define the depth or detail of processing.

        Returns:
        - Tuple containing two DataFrames:
        1. The first DataFrame (`df_clusters`) includes the original texts, their embeddings, and cluster assignments.
        2. The second DataFrame (`df_summary`) contains summaries for each cluster, the specified level of detail,
            and the cluster identifiers.
        """

        # Embed and cluster the texts, resulting in a DataFrame with 'text', 'embd', and 'cluster' columns
        df_clusters = self.embed_cluster_texts(texts)

        # Prepare to expand the DataFrame for easier manipulation of clusters
        expanded_list = []

        # Expand DataFrame entries to document-cluster pairings for straightforward processing
        for index, row in df_clusters.iterrows():
            for cluster in row["cluster"]:
                expanded_list.append(
                    {"text": row["text"], "embd": row["embd"], "cluster": cluster}
                )

        # Create a new DataFrame from the expanded list
        expanded_df = pd.DataFrame(expanded_list)

        # Retrieve unique cluster identifiers for processing
        all_clusters = expanded_df["cluster"].unique()

        print(f"--Generated {len(all_clusters)} clusters--")

        # Summarization
        template = """Here is content from the course DS598: Deep Learning for Data Science.
        The content may be form webapge about the course, or lecture content, or any other relevant information.
        If the content is in bullet points (from  pdf lectre slides), you can summarize the bullet points.
        Give a detailed summary of the content below.
        Documentation:
        {context}
        """
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.model | StrOutputParser()

        # Format text within each cluster for summarization
        summaries = []
        for i in all_clusters:
            df_cluster = expanded_df[expanded_df["cluster"] == i]
            formatted_txt = self.fmt_txt(df_cluster)
            summaries.append(chain.invoke({"context": formatted_txt}))

        # Create a DataFrame to store summaries with their corresponding cluster and level
        df_summary = pd.DataFrame(
            {
                "summaries": summaries,
                "level": [level] * len(summaries),
                "cluster": list(all_clusters),
            }
        )

        return df_clusters, df_summary

    def recursive_embed_cluster_summarize(
        self, texts: List[str], level: int = 1, n_levels: int = 3
    ) -> Dict[int, Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Recursively embeds, clusters, and summarizes texts up to a specified level or until
        the number of unique clusters becomes 1, storing the results at each level.

        Parameters:
        - texts: List[str], texts to be processed.
        - level: int, current recursion level (starts at 1).
        - n_levels: int, maximum depth of recursion.

        Returns:
        - Dict[int, Tuple[pd.DataFrame, pd.DataFrame]], a dictionary where keys are the recursion
        levels and values are tuples containing the clusters DataFrame and summaries DataFrame at that level.
        """
        results = {}  # Dictionary to store results at each level

        # Perform embedding, clustering, and summarization for the current level
        df_clusters, df_summary = self.embed_cluster_summarize_texts(texts, level)

        # Store the results of the current level
        results[level] = (df_clusters, df_summary)

        # Determine if further recursion is possible and meaningful
        unique_clusters = df_summary["cluster"].nunique()
        if level < n_levels and unique_clusters > 1:
            # Use summaries as the input texts for the next level of recursion
            new_texts = df_summary["summaries"].tolist()
            next_level_results = self.recursive_embed_cluster_summarize(
                new_texts, level + 1, n_levels
            )

            # Merge the results from the next level into the current results dictionary
            results.update(next_level_results)

        return results

    def get_vector_db(self):
        """
        Generate a retriever object from a list of documents.

        Parameters:
        - documents: List of document objects.

        Returns:
        - A retriever object.
        """
        leaf_texts = self.split_documents(self.documents)
        results = self.recursive_embed_cluster_summarize(
            leaf_texts, level=1, n_levels=10
        )

        all_texts = leaf_texts.copy()
        # Iterate through the results to extract summaries from each level and add them to all_texts
        for level in sorted(results.keys()):
            # Extract summaries from the current level's DataFrame
            summaries = results[level][1]["summaries"].tolist()
            # Extend all_texts with the summaries from the current level
            all_texts.extend(summaries)

        # Now, use all_texts to build the vectorstore
        vectorstore = FAISS.from_texts(texts=all_texts, embedding=self.embd)
        return vectorstore

    def create_database(self, documents, embedding_model):
        self.documents = documents
        self.embd = embedding_model
        self.vectorstore = self.get_vector_db()
        self.vectorstore.save_local(
            os.path.join(
                self.config["vectorstore"]["db_path"],
                "db_"
                + self.config["vectorstore"]["db_option"]
                + "_"
                + self.config["vectorstore"]["model"],
            )
        )

    def load_database(self, embedding_model):
        self.vectorstore = FAISS.load_local(
            os.path.join(
                self.config["vectorstore"]["db_path"],
                "db_"
                + self.config["vectorstore"]["db_option"]
                + "_"
                + self.config["vectorstore"]["model"],
            ),
            embedding_model,
            allow_dangerous_deserialization=True,
        )
        return self.vectorstore

    def as_retriever(self):
        return self.vectorstore.as_retriever()
