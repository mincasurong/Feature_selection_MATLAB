# Feature Selection MATLAB

This repository provides MATLAB implementations of various feature selection algorithms. These algorithms are essential for preprocessing data in machine learning tasks, helping to identify the most relevant features.

## Contents

- **[Feature_selection_EnsembleLearning.m](Feature_selection_EnsembleLeaning.m)**: Implements feature selection using ensemble learning methods.
- **[Feature_selection_Neuralnetwork.m](Feature_selection_Neuralnetwork.m)**: Implements feature selection using neural network-based methods.

## Feature Selection Algorithms

### Filter Methods

1. **fscchi2**: Univariate feature ranking for classification using chi-square tests.
2. **fscmrmr**: Rank features for classification using the minimum redundancy maximum relevance algorithm.
3. **fsrftest**: Univariate feature ranking for regression using F-tests.
4. **Relieff**: Rank importance of predictors using the ReliefF algorithm.

### Wrapper Methods

1. **sequentialfs**: Sequential feature selection (still updating).

### Embedded Methods

1. **fitcensemble with oobPermutedPredictorImportance**: Embedded method using ensemble learning with out-of-bag permutation predictor importance.

## Getting Started

### Prerequisites

- MATLAB (for running the `.m` files)

### Usage

1. Clone the repository:
    ```bash
    git clone https://github.com/mincasurong/Feature_selection_MATLAB.git
    cd Feature_selection_MATLAB
    ```

2. Open the desired `.m` file in MATLAB and run it to see the results of the feature selection algorithms.

## Reference

This implementation is based on the methods described in various machine learning literature. For more details, please refer to the following resource: [Feature Selection Methods](https://shorturl.at/quBLT).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any improvements or additions.

## Acknowledgements

Special thanks to the MATLAB community and the authors of the feature selection methods for their continuous support and resources.
