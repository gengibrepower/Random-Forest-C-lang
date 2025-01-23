# Random Forest Implementation for Breast Cancer Classification

This project implements a Random Forest algorithm in C for binary classification using a CSV dataset. The dataset used is the `breast-cancer.csv`, which contains information for determining whether a breast cancer diagnosis is positive or negative.You can find the dataset in https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset

## Features

- **Efficient Random Forest implementation** for binary classification.
- Supports **CSV data parsing**.
- Handles a dataset with multiple input variables and 1 target output.
- Outputs **classification results** for each instance in the dataset.
- Achieves an average accuracy of **96%**.

## Prerequisites

To compile and run this project, ensure you have the following:

1. **C Compiler**: GCC or any compatible compiler.
2. **Make** (optional): To simplify the build process.
3. The following libraries (if applicable):
   - `stdlib.h`
   - `stdio.h`
   - `string.h`
   - `time.h`
   - `math.h`

## Dataset Format

The input dataset should be in CSV format with the following structure:

Header:
```csv
id,diagnosis,radius_mean,texture_mean,perimeter_mean,area_mean,smoothness_mean,compactness_mean,concavity_mean,concave points_mean,symmetry_mean,fractal_dimension_mean,radius_se,texture_se,perimeter_se,area_se,smoothness_se,compactness_se,concavity_se,concave points_se,symmetry_se,fractal_dimension_se,radius_worst,texture_worst,perimeter_worst,area_worst,smoothness_worst,compactness_worst,concavity_worst,concave points_worst,symmetry_worst,fractal_dimension_worst
```

- **Input variables**: All columns except `id` and `diagnosis`.
- **Target variable**: The `diagnosis` column, where 1 represents a positive diagnosis, and 0 represents a negative diagnosis.

Example:
```csv
id,diagnosis,radius_mean,texture_mean,perimeter_mean,area_mean,smoothness_mean,compactness_mean,concavity_mean,concave points_mean,symmetry_mean,fractal_dimension_mean,radius_se,texture_se,perimeter_se,area_se,smoothness_se,compactness_se,concavity_se,concave points_se,symmetry_se,fractal_dimension_se,radius_worst,texture_worst,perimeter_worst,area_worst,smoothness_worst,compactness_worst,concavity_worst,concave points_worst,symmetry_worst,fractal_dimension_worst
842302,M,17.99,10.38,122.8,1001,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189
```

## File Structure

- `main.c`: Entry point of the program.
- `treinador.c`: Contains Random Forest training and evaluation implementation.
- `data_loader.c`: Handles reading and parsing CSV files.
- `data_loader.h`: Header file for the project.

## Build and Run

### Using GCC

1. Compile the program using the following command:
   ```bash
   gcc main.c treinador.c data_loader.c -o programa
   ```

2. Run the program:
   ```bash
   ./programa
   ```

## Configuration

The program expects the input CSV file to be named `breast-cancer.csv` and placed in the same directory as the executable. If the file name or location differs, update the file path in the code.

## Output

The program outputs:
- **Number of samples** in the dataset.
- **Number of features** in the dataset.
- **Confusion matrix** off the Random Forest model
- **Accuracy** of the Random Forest model.
- **Prediction Error**
- **Precision**
- **Recall**
- **F1 Score**

Example output:
```plaintext
Numero de amostras: 569
Numero de caracteristicas: 31
Accuracy : 96.49%
Matriz de Confusao:
              Predicao
               P   N
          P | TP | FN |
Real      N | FP | TN |
              Predicao
               P   N
          P | 74 | 1 |
Real      N | 3 | 36 |
Erro de Previsao: 3.51%
Accuracy: 96.49%
Precisao: 97.30%
Recall: 92.31%
F1 Score: 0.95
```

## Limitations

- The dataset must not contain missing or malformed data.
- CSV parsing assumes a standard format with consistent delimiters (`,`).

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Author

Felipe Gegembauer
