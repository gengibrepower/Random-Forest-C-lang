#ifndef DATA_LOADER_H
#define DATA_LOADER_H

// Estrutura para um ponto de dados
typedef struct {
    double *features; // Array com as características
    int label;        // Rótulo (0 ou 1)
} DataPoint;

// Estrutura para um nó da árvore de decisão
typedef struct TreeNode {
    int feature_index;      // qual das caracteristicas esta sendo usada
    double threshold;       // Valor de divisão
    struct TreeNode *left;  // Ponteiro para o filho esquerdo
    struct TreeNode *right; // Ponteiro para o filho direito
    double prediction;      // Previsão se for um nó folha
    int is_leaf;           // Se eh folha ou não (final ou resto da tree)
} TreeNode;

// Estrutura para a floresta de árvores
typedef struct {
    TreeNode **trees; // Array de ponteiros para arvores
    int num_trees;    // Numero de arvores na floresta
} RandomForest;

// Função para carregar dados de um arquivo CSV
DataPoint *load_csv(const char *filename, int *num_samples, int *num_features);

// Funções para manipulacão de trees e forests
TreeNode* create_node(int feature_index, double threshold, int is_leaf);
RandomForest* create_forest(int num_trees);
void free_tree(TreeNode *node);
void free_forest(RandomForest *forest);
int check_homogeneity(DataPoint *data, int num_samples);
int majority_class(DataPoint *data, int num_samples);
int predict(TreeNode *node, DataPoint *data);
int predict_forest(RandomForest *forest, DataPoint *data);

//declara função de treino(estava bugado)
RandomForest* train_forest(DataPoint *data, int num_samples, int num_features, int num_trees, int max_depth);

#endif // DATA_LOADER_H