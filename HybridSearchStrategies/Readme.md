

Combining dense and Sparse Matrices to get better at Retrieval Augumented Generation

Sparse => TF-IDF
Dense  => ( Embeddings + Cosine Similarity )

    Score(hybrid) = alpha * Score(dense) + ( 1 - alpha ) * Score(sparse)

Better RAG model



Key Benefits of Hybrid Search

    1. Boosts Recall

        BM25 catches exact keyword matches

        Semantic search captures meaning even when wording differs

        Together, you reduce the chance of missing relevant documents

    2. Handles Synonyms & Rephrasing

        Semantic search can match queries like “create app” → “build LLM system”
        
        BM25 still catches exact terms like “LLM”, “app”

    3. Improves Retrieval Robustness

        Supports both:

        Users who search with precise keywords

        Users who use natural language questions

    4. Preserves Lexical Importance

        BM25 gives higher weight to rare or critical terms

        Essential for technical, legal, or medical domains
        (e.g., rare terms like “osteoporosis”)

    5. Bridges Document Diversity

        Works well across mixed data sources:

        Web pages

        PDFs

        Blogs

        Well-structured + loosely written text

        Hybrid retrieval adapts better than either method alone

    6. Easy to Tune with Weights

        You can control how much each method influences the final result:

            final_score = 0.7 * semantic_score + 0.3 * bm25_score


        This makes optimization simple and flexible.

    7. More Tolerant to Typos & Variants

        Semantic models handle:

        Misspellings

        Word variants

        BM25 alone may fail here, but hybrid search catches more cases.

Re-Ranking Technique => Hybrid Search Strategy Type

    Second Stage Filtering Process in retrieval systems
    - first use a fast retriever to fetch top-k documents
    - use more accurate but slower model to re-score and re-order those based on the relevancy of the query


Maximal Marginal Relevance

    It selects documents that are both 
    - Relevant to the query
    - Diverse from each other(the documents)
    
    * Prevents the retriever from returning the similar documents that repeat the same content

    MMR(doc) = lambda * Sim( doc , query ) - ( 1 - lambda ) * max(s belongs to S) Sim( doc , S)
    
    the MMR(doc) should be high which means it is diverse and also relevant 
    
    if not clear do with an example

When to use MMR :
    . In a RAG to avoid feeding the LLM with redundant documents 
    . ChatBots : FAQ , Search APP 
    . Retriever already returns many results 
    . MMR + Hybrid retriever => Deadly combo ( i think )

When not to use :
    . You are already re-ranking 
    . Documents are already diverse 
    . Extreme short context window
    . You may want top 1 relevant 
    . You need precision only
