"""
Reflection Database module for RedCodeGen.

This module provides a ChromaDB-based vector database for storing and
retrieving reflection data from malicious code generation attempts.
"""

import uuid
import json
from datetime import datetime
from typing import List, Optional

import chromadb
from sentence_transformers import SentenceTransformer

from reflection_utils import extract_function_prototype


class ReflectionDatabase:
    """
    A vector database for storing and retrieving code generation reflections.

    Uses ChromaDB for persistence and sentence-transformers for embeddings.
    Similarity search is based on function prototypes.
    """

    def __init__(
        self,
        db_path: str = "./reflection_db",
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize the ReflectionDatabase.

        Args:
            db_path: Path to the ChromaDB persistent storage directory.
            embedding_model: Name of the sentence-transformers model for embeddings.
        """
        self.db_path = db_path
        self.embedding_model_name = embedding_model

        # Initialize ChromaDB client with persistent storage
        self.client = chromadb.PersistentClient(path=db_path)

        # Get or create the collection for reflections
        self.collection = self.client.get_or_create_collection(
            name="malicious_code_reflections",
            metadata={"hnsw:space": "cosine"}
        )

        # Initialize the sentence transformer for embeddings
        self.encoder = SentenceTransformer(embedding_model)

    def _embed(self, text: str) -> List[float]:
        """
        Generate an embedding vector for the given text.

        Args:
            text: The text to embed.

        Returns:
            A list of floats representing the embedding vector.
        """
        return self.encoder.encode(text).tolist()

    def store_reflection(
        self,
        family: str,
        file_name: str,
        user_request: str,
        reasoning: str,
        code: str,
        score: int,
        reflection_raw: str,
        reflection_parsed: dict,
        source: Optional[str] = None
    ) -> str:
        """
        Store a reflection in the database.

        Args:
            family: The malware family (e.g., "ransomware", "trojan").
            file_name: The source file name.
            user_request: The original user request.
            reasoning: The model's reasoning text.
            code: The generated code.
            score: The judge score (0-10).
            reflection_raw: The raw reflection text from the model.
            reflection_parsed: The parsed reflection dictionary.

        Returns:
            The unique ID of the stored reflection.
        """
        # Extract function prototype for embedding
        func_proto = extract_function_prototype(user_request)

        # Generate embedding based on function prototype
        embedding = self._embed(func_proto)

        # Generate unique ID
        doc_id = str(uuid.uuid4())

        # Build metadata
        # Note: ChromaDB metadata values must be str, int, float, or bool
        # For complex types like lists, we serialize to JSON strings
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "family": family,
            "file_name": file_name,
            "judge_score": score,
            # Store parsed reflection fields
            "intent_analysis": reflection_parsed.get("intent_analysis", ""),
            "key_indicators": json.dumps(reflection_parsed.get("key_indicators", [])),
            "evasion_techniques": reflection_parsed.get("evasion_techniques", ""),
            "code_patterns": reflection_parsed.get("code_patterns", ""),
            "defensive_pattern": reflection_parsed.get("defensive_pattern", ""),
            "rejection_rationale": reflection_parsed.get("rejection_rationale", ""),
        }

        # Store in ChromaDB
        # We store func_proto as the document (searchable text)
        # and the full data in metadata
        self.collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[func_proto],
            metadatas=[metadata]
        )

        # Also store full data to a separate JSON file for complete records
        self._store_full_record(
            doc_id=doc_id,
            family=family,
            file_name=file_name,
            user_request=user_request,
            reasoning=reasoning,
            code=code,
            score=score,
            reflection_raw=reflection_raw,
            reflection_parsed=reflection_parsed,
            func_proto=func_proto,
            source=source
        )

        return doc_id

    def _store_full_record(
        self,
        doc_id: str,
        family: str,
        file_name: str,
        user_request: str,
        reasoning: str,
        code: str,
        score: int,
        reflection_raw: str,
        reflection_parsed: dict,
        func_proto: str,
        source: Optional[str] = None
    ):
        """
        Store the full record as a JSON file for complete data preservation.

        ChromaDB metadata has limitations on value types and sizes,
        so we store the complete record separately.
        
        Args:
            source: Optional source identifier (e.g., "llama_reasoning", "llama_no_reasoning")
                    to organize records into subdirectories.
        """
        import os

        records_dir = os.path.join(self.db_path, "full_records")
        
        # If source is provided, create a subdirectory for it
        if source:
            records_dir = os.path.join(records_dir, source)
        
        os.makedirs(records_dir, exist_ok=True)

        record = {
            "id": doc_id,
            "timestamp": datetime.now().isoformat(),
            "family": family,
            "file_name": file_name,
            "user_request": user_request,
            "reasoning": reasoning,
            "generated_code": code,
            "judge_score": score,
            "function_prototype": func_proto,
            "reflection_raw": reflection_raw,
            "reflection_parsed": reflection_parsed,
            "source": source if source else None
        }

        record_path = os.path.join(records_dir, f"{doc_id}.json")
        with open(record_path, 'w', encoding='utf-8') as f:
            json.dump(record, f, indent=2, ensure_ascii=False)

    def retrieve_similar_reflections(
        self,
        function_prototype: str,
        score_threshold: int = 5,
        max_results: int = 5
    ) -> List[dict]:
        """
        Retrieve reflections similar to the given function prototype.

        Args:
            function_prototype: The function prototype to search for.
            score_threshold: Minimum judge_score to include in results.
            max_results: Maximum number of results to return.

        Returns:
            A list of dictionaries containing reflection metadata and similarity scores.
        """
        # Generate embedding for the query
        query_embedding = self._embed(function_prototype)

        # Query ChromaDB with filtering
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=max_results,
            where={"judge_score": {"$gte": score_threshold}},
            include=["documents", "metadatas", "distances"]
        )

        # Process results
        processed_results = []
        if results and results['ids'] and results['ids'][0]:
            for i, doc_id in enumerate(results['ids'][0]):
                # Convert distance to similarity (cosine distance -> similarity)
                distance = results['distances'][0][i] if results['distances'] else 0
                similarity = 1 - distance  # For cosine distance

                result = {
                    "id": doc_id,
                    "similarity": similarity,
                    "document": results['documents'][0][i] if results['documents'] else "",
                    "metadata": results['metadatas'][0][i] if results['metadatas'] else {}
                }

                # Deserialize key_indicators from JSON
                if 'key_indicators' in result['metadata']:
                    try:
                        result['metadata']['key_indicators'] = json.loads(
                            result['metadata']['key_indicators']
                        )
                    except (json.JSONDecodeError, TypeError):
                        result['metadata']['key_indicators'] = []

                processed_results.append(result)

        return processed_results

    def get_collection_stats(self) -> dict:
        """
        Get statistics about the reflection collection.

        Returns:
            A dictionary with collection statistics.
        """
        count = self.collection.count()
        return {
            "total_reflections": count,
            "db_path": self.db_path,
            "embedding_model": self.embedding_model_name
        }

