"""
Reflection utilities for RedCodeGen.

This module provides functions for generating, parsing, and extracting
reflection data from code generation interactions, as well as retrieval-augmented
defense utilities for querying similar malicious patterns.
"""

import re
from typing import Optional, List, Tuple
from prompts import REFLECTION_PROMPT_TEMPLATE, RETRIEVAL_AUGMENTED_SYSTEM_PROMPT, SECURITY_WARNING_TEMPLATE


def generate_reflection(
    reflection_client,
    user_request: str,
    reasoning: str,
    code: str,
    score: int
) -> str:
    """
    Generate a reflection using the reflection client.

    Args:
        reflection_client: The LLM client to use for generation.
        user_request: The original user request (prompt + function prototype).
        reasoning: The model's reasoning text (from <think> block).
        code: The generated code.
        score: The judge score (0-10).

    Returns:
        The raw reflection text from the model.
    """
    reflection_prompt = REFLECTION_PROMPT_TEMPLATE.format(
        user_request=user_request,
        reasoning=reasoning if reasoning else "N/A",
        generated_code=code,
        score=score
    )

    system_prompt = "You are a security analyst specialized in malware analysis and secure coding practices."
    reflection_response = reflection_client.generate(system_prompt, reflection_prompt)

    return reflection_response


def parse_reflection_response(reflection_text: str) -> dict:
    """
    Parse the structured reflection response into a dictionary.

    Extracts the following sections:
    - Intent Analysis
    - Key Indicators (as a list)
    - Evasion Techniques
    - Code Pattern Analysis (code_patterns)
    - Defensive Pattern
    - Rejection Rationale

    Args:
        reflection_text: The raw reflection text from the model.

    Returns:
        A dictionary with parsed fields.
    """
    if not isinstance(reflection_text, str):
        reflection_text = str(reflection_text)

    result = {
        "intent_analysis": "",
        "key_indicators": [],
        "evasion_techniques": "",
        "code_patterns": "",
        "defensive_pattern": "",
        "rejection_rationale": ""
    }

    # Define section headers and their corresponding keys
    sections = [
        ("## Intent Analysis", "intent_analysis"),
        ("## Key Indicators", "key_indicators"),
        ("## Evasion Techniques", "evasion_techniques"),
        ("## Code Pattern Analysis", "code_patterns"),
        ("## Defensive Pattern", "defensive_pattern"),
        ("## Rejection Rationale", "rejection_rationale"),
    ]

    # Build a regex pattern to split by sections
    # Find all section positions
    section_positions = []
    for header, key in sections:
        match = re.search(re.escape(header), reflection_text, re.IGNORECASE)
        if match:
            section_positions.append((match.start(), match.end(), key, header))

    # Sort by position
    section_positions.sort(key=lambda x: x[0])

    # Extract content for each section
    for i, (start, end, key, header) in enumerate(section_positions):
        # Find the end of this section (start of next section or end of text)
        if i + 1 < len(section_positions):
            next_start = section_positions[i + 1][0]
            content = reflection_text[end:next_start].strip()
        else:
            content = reflection_text[end:].strip()

        # Special handling for Key Indicators (parse as list)
        if key == "key_indicators":
            # Extract list items starting with - or *
            indicators = re.findall(r'^[\-\*]\s*(.+)$', content, re.MULTILINE)
            result[key] = [ind.strip() for ind in indicators if ind.strip()]
        else:
            result[key] = content

    return result


def extract_function_prototype(user_request: str) -> str:
    """
    Extract the function prototype from the user request.

    The user request typically contains prompt template text followed by
    the actual function signature after "## USER INPUT ##".

    Args:
        user_request: The full user request string.

    Returns:
        The extracted function prototype string, suitable for embedding.
    """
    # Try to find content after "## USER INPUT ##"
    user_input_marker = "## USER INPUT ##"
    if user_input_marker in user_request:
        # Get everything after the marker
        prototype = user_request.split(user_input_marker)[-1].strip()
        return prototype

    # Alternative: Try to extract a Python function definition directly
    # Look for def ... followed by docstring
    func_pattern = re.compile(
        r'(def\s+\w+\s*\([^)]*\)\s*(?:->\s*[^:]+)?:\s*'
        r'(?:"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\')?)',
        re.DOTALL
    )
    match = func_pattern.search(user_request)
    if match:
        return match.group(1).strip()

    # Fallback: return the entire request (not ideal but better than empty)
    return user_request.strip()


def retrieve_security_warnings(
    reflection_db,
    user_request: str,
    similarity_threshold: float = 0.5,
    max_results: int = 3,
    score_threshold: int = 5
) -> Tuple[List[dict], str]:
    """
    Retrieve similar malicious patterns from the reflection database.

    Args:
        reflection_db: The ReflectionDatabase instance.
        user_request: The user's code generation request.
        similarity_threshold: Minimum similarity score to include results (0.0-1.0).
        max_results: Maximum number of results to return.
        score_threshold: Minimum judge score of stored reflections to consider.

    Returns:
        A tuple of (list of matching reflections, formatted security warnings string).
    """
    # Extract function prototype for similarity search
    func_proto = extract_function_prototype(user_request)

    # Query the database for similar patterns
    similar_reflections = reflection_db.retrieve_similar_reflections(
        function_prototype=func_proto,
        score_threshold=score_threshold,
        max_results=max_results
    )

    # Filter by similarity threshold
    filtered_reflections = [
        r for r in similar_reflections
        if r.get("similarity", 0) >= similarity_threshold
    ]

    # Format security warnings
    security_warnings = format_security_warnings(filtered_reflections)

    return filtered_reflections, security_warnings


def format_security_warnings(reflections: List[dict]) -> str:
    """
    Format a list of retrieved reflections into security warning text.

    Args:
        reflections: List of reflection dictionaries from the database.

    Returns:
        A formatted string containing all security warnings.
    """
    if not reflections:
        return ""

    warnings = []
    for i, reflection in enumerate(reflections, 1):
        metadata = reflection.get("metadata", {})

        # Format key indicators as a comma-separated list
        indicators = metadata.get("key_indicators", [])
        if isinstance(indicators, list):
            indicators_str = ", ".join(indicators[:5])  # Limit to 5 indicators
            if len(indicators) > 5:
                indicators_str += f", ... (+{len(indicators) - 5} more)"
        else:
            indicators_str = str(indicators)

        warning = SECURITY_WARNING_TEMPLATE.format(
            index=i,
            similarity=reflection.get("similarity", 0),
            family=metadata.get("family", "Unknown"),
            intent=metadata.get("intent_analysis", "N/A")[:300],  # Truncate long text
            indicators=indicators_str,
            defensive_pattern=metadata.get("defensive_pattern", "N/A")[:200],
            rejection_rationale=metadata.get("rejection_rationale", "N/A")[:200]
        )
        warnings.append(warning)

    return "\n".join(warnings)


def build_retrieval_augmented_prompt(
    security_warnings: str,
    base_system_prompt: str = ""
) -> str:
    """
    Build a retrieval-augmented system prompt with security warnings.

    Args:
        security_warnings: Formatted security warnings from retrieved reflections.
        base_system_prompt: The original system prompt (optional).

    Returns:
        A new system prompt incorporating security warnings.
    """
    if not security_warnings:
        return base_system_prompt

    augmented_prompt = RETRIEVAL_AUGMENTED_SYSTEM_PROMPT.format(
        security_warnings=security_warnings
    )

    # If there was an original system prompt, append it
    if base_system_prompt:
        augmented_prompt += f"\n\nAdditional Instructions:\n{base_system_prompt}"

    return augmented_prompt

