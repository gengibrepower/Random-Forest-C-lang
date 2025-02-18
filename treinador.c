#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "data_loader.h"

//Felipe Gegembauer

// Função para calcular o Gini Impurity de um conjunto de dados(opção 1)
double calculate_gini_impurity(DataPoint *data, int num_samples) {
    int count_0 = 0, count_1 = 0;
    for (int i = 0; i < num_samples; i++) {
        if (data[i].label == 0) {
            count_0++;
        } else {
            count_1++;
        }
    }
    double p_0 = (double)count_0 / num_samples;
    double p_1 = (double)count_1 / num_samples;
    return 1.0 - (p_0 * p_0 + p_1 * p_1); // Fórmula do Gini Impurity
}

//Função para caucular a entropia de um conunto de dados(opção 2)
double calculate_entropy(DataPoint *data, int num_samples) {
    int count_0 = 0, count_1 = 0;
    for (int i = 0; i < num_samples; i++) {
        if (data[i].label == 0) {
            count_0++;
        } else {
            count_1++;
        }
    }
    double p_0 = (double)count_0 / num_samples;
    double p_1 = (double)count_1 / num_samples;

    double entropy = 0.0;
    if (p_0 > 0) {
        entropy -= p_0 * log2(p_0);
    }
    if (p_1 > 0) {
        entropy -= p_1 * log2(p_1);
    }
    return entropy;
}


int compare_feature(const void *a, const void *b) {
    DataPoint *data_a = (DataPoint *)a;
    DataPoint *data_b = (DataPoint *)b;
    // Compara a característica 0 ou outra, dependendo de qual se deseja usar
    return (data_a->features[0] > data_b->features[0]) - (data_a->features[0] < data_b->features[0]);
}

// Função para filtrar os dados para o grupo esquerdo
DataPoint* filter_left_data(DataPoint *data, int num_samples, int best_feature, double best_threshold, int *left_size) {
    DataPoint *left_data = malloc(num_samples * sizeof(DataPoint)); // Aloca um espaço maior, será ajustado depois
    *left_size = 0;

    for (int i = 0; i < num_samples; i++) {
        if (data[i].features[best_feature] <= best_threshold) {
            left_data[(*left_size)++] = data[i]; // Adiciona ao grupo esquerdo
        }
    }
    return left_data;
}

// Função para filtrar os dados para o grupo direito
DataPoint* filter_right_data(DataPoint *data, int num_samples, int best_feature, double best_threshold, int *right_size) {
    DataPoint *right_data = malloc(num_samples * sizeof(DataPoint)); // Aloca um espaço maior, será ajustado depois
    *right_size = 0;

    for (int i = 0; i < num_samples; i++) {
        if (data[i].features[best_feature] > best_threshold) {
            right_data[(*right_size)++] = data[i]; // Adiciona ao grupo direito
        }
    }
    return right_data;
}

//usa gini
void find_best_split(DataPoint *data, int num_samples, int num_features, int *best_feature, double *best_threshold) {
    double best_gini = 1.0; // Melhor Gini inicialmente o pior valor possível
    *best_feature = -1;
    *best_threshold = 0.0;

    for (int feature = 0; feature < num_features; feature++) {
        // Ordena os dados pela característica 'feature'
        qsort(data, num_samples, sizeof(DataPoint), compare_feature);  // Ordena os dados pela característica

        for (int i = 0; i < num_samples - 1; i++) {
            // Pega o valor médio entre dois pontos consecutivos como o threshold
            double threshold = (data[i].features[feature] + data[i + 1].features[feature]) / 2.0;

            // Contagem de elementos para os grupos esquerdo e direito
            int left_size = 0, right_size = 0;
            for (int j = 0; j < num_samples; j++) {
                if (data[j].features[feature] <= threshold) {
                    left_size++;
                } else {
                    right_size++;
                }
            }
            // Divida os dados em dois grupos (esquerdo e direito)
            DataPoint *left_data = malloc(left_size * sizeof(DataPoint));
            DataPoint *right_data = malloc(right_size * sizeof(DataPoint));

            int left_index = 0, right_index = 0;
            for (int j = 0; j < num_samples; j++) {
                if (data[j].features[feature] <= threshold) {
                    left_data[left_index++] = data[j];
                } else {
                    right_data[right_index++] = data[j];
                }
            }

            // Calcular a impureza de Gini para os dois grupos
            double gini_left = calculate_gini_impurity(left_data, left_size);
            double gini_right = calculate_gini_impurity(right_data, right_size);

            // Calcula a impureza ponderada
            double gini = (left_size / (double)num_samples) * gini_left + (right_size / (double)num_samples) * gini_right;

            // Atualiza o melhor threshold e feature se a impureza for menor
            if (gini < best_gini) {
                best_gini = gini;
                *best_feature = feature;
                *best_threshold = threshold;
            }

            // Libera memória dos dados divididos
            free(left_data);
            free(right_data);
        }
    }
}


//usa entropia
void find_best_split_entropy(DataPoint *data, int num_samples, int num_features, int *best_feature, double *best_threshold) {
    double best_entropy = INFINITY; // Melhor entropia inicialmente o pior valor possível
    *best_feature = -1;
    *best_threshold = 0.0;

    for (int feature = 0; feature < num_features; feature++) {
        // Ordena os dados pela característica 'feature'
        qsort(data, num_samples, sizeof(DataPoint), compare_feature);  // Ordena os dados pela característica

        for (int i = 0; i < num_samples - 1; i++) {
            // Pega o valor médio entre dois pontos consecutivos como o threshold
            double threshold = (data[i].features[feature] + data[i + 1].features[feature]) / 2.0;

            // Contagem de elementos para os grupos esquerdo e direito
            int left_size = 0, right_size = 0;
            for (int j = 0; j < num_samples; j++) {
                if (data[j].features[feature] <= threshold) {
                    left_size++;
                } else {
                    right_size++;
                }
            }
            // Divida os dados em dois grupos (esquerdo e direito)
            DataPoint *left_data = malloc(left_size * sizeof(DataPoint));
            DataPoint *right_data = malloc(right_size * sizeof(DataPoint));

            int left_index = 0, right_index = 0;
            for (int j = 0; j < num_samples; j++) {
                if (data[j].features[feature] <= threshold) {
                    left_data[left_index++] = data[j];
                } else {
                    right_data[right_index++] = data[j];
                }
            }

            // Calcular a entropia para os dois grupos
            double entropy_left = calculate_entropy(left_data, left_size);
            double entropy_right = calculate_entropy(right_data, right_size);

            // Calcula a entropia ponderada
            double entropy = (left_size / (double)num_samples) * entropy_left + (right_size / (double)num_samples) * entropy_right;

            // Atualiza o melhor threshold e feature se a entropia for menor
            if (entropy < best_entropy) {
                best_entropy = entropy;
                *best_feature = feature;
                *best_threshold = threshold;
            }

            // Libera memória dos dados divididos
            free(left_data);
            free(right_data);
        }
    }
}

//Função para treinar arvore de decisão
TreeNode* train_tree(DataPoint *data, int num_samples, int num_features, int max_depth) {
    // Caso base: Se a profundidade máxima for atingida ou todos os dados são do mesmo rótulo
    if (max_depth == 0 || check_homogeneity(data, num_samples)) {
        // Aqui podemos retornar um nó folha com o rótulo mais comum
        TreeNode *leaf = create_node(-1, 0.0, 1);
        leaf->prediction = majority_class(data, num_samples);
        return leaf;
    }

    int best_feature;
    double best_threshold;
    find_best_split_entropy(data, num_samples, num_features, &best_feature, &best_threshold);

    TreeNode *node = create_node(best_feature, best_threshold, 0); // Não é folha
    int left_size, right_size;
    
    // Dividir os dados em dois conjuntos com base no threshold
    DataPoint *left_data = filter_left_data(data, num_samples, best_feature, best_threshold, &left_size);
    DataPoint *right_data = filter_right_data(data, num_samples, best_feature, best_threshold, &right_size);

    // Recursivamente treinar as subárvores
    node->left = train_tree(left_data, left_size, num_features, max_depth - 1);
    node->right = train_tree(right_data, right_size, num_features, max_depth - 1);

    // Libera a memória alocada para os dados divididos
    free(left_data);
    free(right_data);

    return node;
}

//verifica se todos os dados em um nó são do mesmo rótulo
int check_homogeneity(DataPoint *data, int num_samples) {
    int first_label = data[0].label;
    for (int i = 1; i < num_samples; i++) {
        if (data[i].label != first_label) {
            return 0; // Não homogêneo
        }
    }
    return 1; // Homogêneo
}


//retorna o rótulo mais comum em um conjunto de dados
int majority_class(DataPoint *data, int num_samples) {
    int count_0 = 0, count_1 = 0;
    for (int i = 0; i < num_samples; i++) {
        if (data[i].label == 0) {
            count_0++;
        } else {
            count_1++;
        }
    }
    return count_0 > count_1 ? 0 : 1;
}

//cria uma amostra com reposição para o treinamento de cada árvore na floresta
DataPoint* bootstrap_sample(DataPoint *data, int num_samples) {
    DataPoint *sample = malloc(num_samples * sizeof(DataPoint));
    for (int i = 0; i < num_samples; i++) {
        int index = rand() % num_samples;  // Gera um índice aleatório
        sample[i] = data[index];  // Copia o ponto de dados para a amostra
    }
    return sample;
}


//Função para treinar floresta
RandomForest* train_forest(DataPoint *data, int num_samples, int num_features, int num_trees, int max_depth) {
    RandomForest *forest = create_forest(num_trees);
    for (int i = 0; i < num_trees; i++) {
        // Amostragem com reposição
        DataPoint *bootstrap_data = bootstrap_sample(data, num_samples);
        int bootstrap_size = num_samples; // O tamanho do bootstrap é igual ao número de amostras originais
        forest->trees[i] = train_tree(bootstrap_data, bootstrap_size, num_features, max_depth);
        free(bootstrap_data);
    }
    return forest;
}
// Função para prever usando uma árvore de decisão
int predict(TreeNode *node, DataPoint *data) {
    if (node->is_leaf) {
        return (int)node->prediction; // Retorna a previsão se for um nó folha
    }

    // Verifica em qual ramo seguir
    if (data->features[node->feature_index] <= node->threshold) {
        return predict(node->left, data); // Segue para o filho esquerdo
    } else {
        return predict(node->right, data); // Segue para o filho direito
    }
}

// Função para prever usando a floresta aleatória
int predict_forest(RandomForest *forest, DataPoint *data) {
    int count_0 = 0, count_1 = 0;

    for (int i = 0; i < forest->num_trees; i++) {
        int prediction = predict(forest->trees[i], data);
        if (prediction == 0) {
            count_0++;
        } else {
            count_1++;
        }
    }

    return (count_1 > count_0) ? 1 : 0; // Retorna a classe mais comum
}