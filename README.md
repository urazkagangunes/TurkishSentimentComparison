# Turkish Sentiment Comparison

Comparative evaluation of lexicon-based, dependency-aware,
and LLM-based sentiment analysis approaches for Turkish.

## Experimental Settings

The project evaluates four configurations:

1. HisNet
2. HisNet + Dependency Parsing
3. LLM
4. LLM + Dependency Parsing

Stanford Stanza is used to obtain Turkish dependency parses.

## Project Structure

- `data/`: Raw and processed sentiment datasets
- `src/hisnet/`: HisNet-based sentiment methods
- `src/dependency/`: Stanford Stanza dependency parsing
- `src/llm/`: LLM-based sentiment classification
- `src/evaluation/`: Evaluation metrics and comparison
- `prompts/`: LLM prompts
- `results/`: Experimental results
- `tests/`: Tests

## Dependency Parser

Turkish dependency parsing is performed using Stanford Stanza.

Processors:

- Tokenization
- Multi-word token expansion
- POS tagging
- Lemmatization
- Dependency parsing