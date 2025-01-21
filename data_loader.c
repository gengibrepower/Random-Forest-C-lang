#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "data_loader.h"

//Felipe Gegembauer

//Funcao para carregar o csv
DataPoint *load_csv(const char *filename, int *num_samples, int *num_features) {
    FILE *arquivo = fopen(filename, "r"); //abre a bomba do arquivo como "arquivo"
    if (!arquivo) {
        perror("não consegui abrir o arquivo");
        return NULL;
    }

    char linha[1024];                   //vou armazenar o conteudo da linha com + ou - 1024 caracteres
    int samples = 0, features = 0;      // features eh o numero de atributos relevantes sample eh o numero de linhas de dados

    //um contador de linhas e colunas
    while (fgets(linha, sizeof(linha), arquivo)) {
        if (samples == 0) {
            char *token = strtok(linha, ",");
            while (token) {
                features++;
                token = strtok(NULL, ",");  //toda vez que isso aparecer eh como se estivesse ignorando itens nulos e pulando para o prox
            }
            features--;
        }
        samples++;
    }

    //abrir espaço na memória
    DataPoint *data = malloc(samples * sizeof(DataPoint));
    for(int i=0; i<samples; i++){
        data[i].features = malloc(features * sizeof(double));
    }

    rewind(arquivo);    //vo voltar para o inicio do arquivo
    fgets(linha, sizeof(linha), arquivo); // sem contar o cabeçalho
    int fileira = 0;

    while (fgets(linha, sizeof(linha), arquivo)){
        char *token = strtok(linha, ",");   //cortando o csv pelas virgulas
        token = strtok(NULL, ",");

        data[fileira].label = (token[0] == 'M') ? 1 : 0; //troca os diagnosticos por 1 e 0 para serem melhor tratados
        token = strtok(NULL, ",");

        int coluna = 0;
        while (token){
            data[fileira].features[coluna++] = atof(token); //string para double(float buffado)
            token = strtok(NULL, ",");
        }

        fileira++;
    }

    fclose(arquivo);
    *num_samples = samples - 1; //sempre vou ignorar o cabecalho
    *num_features = features;
    return data;
}

// Funcão para um nó
TreeNode* create_node(int feature_index, double threshold, int is_leaf) {
    TreeNode *node = (TreeNode *)malloc(sizeof(TreeNode));
    node->feature_index = feature_index;
    node->threshold = threshold;
    node->left = NULL;
    node->right = NULL;
    node->prediction = 0.0; // Inicializa a previsão
    node->is_leaf = is_leaf; // Define se é um nó folha
    return node;
}

// Funcão para uma forest
RandomForest* create_forest(int num_trees) {
    RandomForest *forest = (RandomForest *)malloc(sizeof(RandomForest));
    forest->num_trees = num_trees;
    forest->trees = (TreeNode **)malloc(num_trees * sizeof(TreeNode *));
    return forest;
}

// Função para liberar a memória da árvore
void free_tree(TreeNode *node) {
    if (node) {
        free_tree(node->left);
        free_tree(node->right);
        free(node);
    }
}

// Função para liberar a memória da floresta
void free_forest(RandomForest *forest) {
    for (int i = 0; i < forest->num_trees; i++) {
        free_tree(forest->trees[i]);
    }
    free(forest->trees);
    free(forest);
}