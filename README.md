# Anticipatory Learning Classifier 2 with Value Consistency Prioritization implemented in Python

Repository contains implementation of agents from the family of Learning Classifier Systems such as ACS or ACS2, enhanced with techniques like Experience Replay, Hindsight Experience Replay and a new technique - Value Consistency Prioritization.

Value Consistency Prioritization is based on the paper by Claudia Russo, Daniela Barni, Ioana Zagrean and Francesca Danioni:
https://www.mdpi.com/1210682

## Repository Structure

This repository contains:
- **openai-envs** - Gymnasium library with added new environments ([Gymnasium](https://gymnasium.farama.org))
- **pyalcs** - LCS agents with new ACS2VCP agent ([Documentation](https://pyalcs.readthedocs.io/en/latest/), [GitHub](https://github.com/ParrotPrediction/pyalcs))
- **pyalcs-experiments** - Scripts and notebooks with experiments

## Development

Create conda environment:
```bash
cd pyalcs-experiments
conda env create --file environment-base.yml
conda activate pyalcs-experiments
conda env update --file environment-base.yml --prune
```

## Run Scripts

```bash
export PYTHONPATH=/path/to/pyalcs-experiments
python ../pyalcs-experiments/scripts/ACS2VCP/run_acs2vcp_maze4.py
```

