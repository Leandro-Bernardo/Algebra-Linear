import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional

class EmbeddingTaxonomico(nn.Module):
    """
        Constroi uma tabela de consulta (lookup table) aprendida que armazena vetores contextuais (embeddings) taxonomicos.
        Cada nó da árvore será representado por um vetor contextual.
        Ou seja: cada nó de cada um dos possíveis niveis hierárquicos possuirá uma representação semântica aprendida.
        Args:
            num_embeddings: quantidade de nós na árvore de taxonomia
            embedding_dim: tamanho de cada vetor contextual taxonomico
        Returns:
            uma matriz de tamanho 6 x (embedding_dim)
    """
    def __init__(self, num_total_nos_globais: int, embedding_dim: Optional[int] = 49):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings = num_total_nos_globais, embedding_dim = embedding_dim)

        # self.mlp_contexto = nn.Sequential(
        #                                 nn.Linear(in_features=6 * embedding_dim, out_features=128),
        #                                 nn.ReLU(),
        #                                 nn.Linear(in_features=128, out_features=6 * embedding_dim),)

        self.embedding_dim = embedding_dim

    def forward(self, x):
        # x: [Batch, 6] (IDs inteiros)
        # Ex de x[0]: tensor([2, 14, 52, 102, 4, 9]) onde: Reino = 2, Filo = 14, Ordem = 52, Classe = 102, Familia 4, Genero = 9
        batch_size = x.size(0)

        # Conlsulta a tabela para obter os embeddings: [Batch, 6] -> [Batch, 6, 49]
        embedded = self.embedding(x)

        # Achata os vetores para que a MLP possa cruzar as informações de todos os níveis
        # flat_embedded = embedded.view(batch_size, -1) # [Batch, 343]

        # Ao passar pela MLP, ocorre a contextualização entre diferentes níveis. O Reino, o Filo, a Ordem, etc., são misturados e geram novas representações dependentes umas das outras.
        # contextualizado = self.mlp_contexto(flat_embedded) # [Batch, 343]

        # Modela de volta no formato da matriz 6 x 49
        # matriz_final = contextualizado.view(batch_size, 6, self.embedding_dim) # [Batch, 6, 49]

        return embedded#matriz_final


class FeatureExtractor(nn.Module):
    """
        Modelo extrator de caracteristica para processar as imagens
        Args:
        Returns: features extraidas das camadas convolucionais
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        self.features = model.features
        # congela os pesos da VGG, desconectando-a do grafo
        for p in self.features.parameters():
            p.requires_grad = False

    def forward(self, x ):
        x = self.features(x)
        return x

class Model(nn.Module):
    """
        Modelo end-to-end para classificação de espécies, seguindo a taxonomia hierárquica.
        Args:
            num_embeddings: quantidade de nós na árvore de taxonomia
            embedding_dim: tamanho de cada vetor contextual taxonomico
        Returns: classe predita

    """
    def __init__(self, num_classes_especie: int, num_total_nos_globais: int, embedding_dim: Optional[int] = 49):
        super().__init__()
        self.feature_extractor = FeatureExtractor()
        self.embedding_taxonomico = EmbeddingTaxonomico(num_total_nos_globais, embedding_dim)
        self.reducao_canais = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1) # Projeta os 512 canais da VGG para 1 canal sem perder informação espacial
        self.pool_adaptativo = nn.AdaptiveAvgPool2d((7, 7)) # Força dimensão espacial 7x7 = 49
        self.classificador = torch.nn.Sequential(
                                               nn.Linear(in_features=294, out_features=256),
                                               nn.BatchNorm1d(256),
                                               nn.ReLU(),
                                               nn.Linear(in_features=256, out_features=128),
                                               nn.BatchNorm1d(128),
                                               nn.ReLU(),
                                               nn.Linear(in_features=128, out_features=64),
                                               nn.BatchNorm1d(64),
                                               nn.ReLU(),
                                               nn.Linear(in_features=64, out_features=32),
                                               nn.BatchNorm1d(32),
                                               nn.ReLU(),
                                               nn.Linear(in_features=32, out_features=num_classes_especie),
                                             )

    def forward(self, images: torch.Tensor, IDs_taxonomicos: torch.Tensor):
        ## PROCESSAMENTO DA IMAGEM
        # Saída da VGG
        features  = self.feature_extractor(images) # [Batch, 512, H, W]
        features_reduzidas = self.reducao_canais(features) # [Batch, 1, H, W]
        features_espaciais = self.pool_adaptativo(features_reduzidas) # [Batch, 1, 7, 7]
            # Faz uma média entre os canais para resumir os mapas em um só
            #mean = torch.mean(features, dim=1, keepdim=True)  # [Batch, 1, 6, 6]
            # Faz uma media adaptativa entre os canais para resumir os mapas
            #mean = torch.nn.AdaptiveAvgPool2d((1,1)).squeeze() # [Batch, 512, 1, 1]
        # Faz um flattening para gerar um vetor de catacteristicas semanticas
        vetor_visual = torch.flatten(features_espaciais, 1) # [Batch, 49]
        # adiciona uma dimensão extra para o broadcasting resolver o produto de Hamard
        vetor_visual = vetor_visual.unsqueeze(1) # [Batch, 1, 49]

        ## PROCESSAMENTO DA TAXONOMIA
        matriz_tax = self.embedding_taxonomico(IDs_taxonomicos) # [Batch, 6, 49]

        ## IMBUTE AS INFORMAÇÕES TAXONOMICAS APRENDIDAS COM AS INFORMAÇõES VISUAIS APRENDIDAS
        # Produto de Hadamard
        features = vetor_visual*matriz_tax # [Batch, 6, 49]
        # Faz um flattening para gerar um vetor de catacteristicas semanticas e hierárquicas, que será a entrada da rede classificadora
        features_flattened = torch.flatten(features, 1) # [Batch, 294]

        ## SVD
        # adiciona ruído infinitesimal apenas para que as matrizes nunca possuam posto degenerado puro
        eps_svd = 1e-6 * torch.randn_like(features)
        # Calcula SVD para a matriz de caracteristicas semanticas hierarquicas
        # (torch.linalg.svdvals: ~supports batches of matrices, and if A is a batch of matrices then the output has the same batch dimensions~)
        sigmas = torch.linalg.svdvals(features + eps_svd)

        ## CLASSIFICAÇÃO FINAL
        # A rede usa as características extraídas para prever a Espécie
        saida_especie = self.classificador(features_flattened)

        # Retorna a *saída predita* para Loss Cross Entropy; a *matriz de caracteristicas semanticas e hierárquicas* e os *sigmas* para a Loss Taxonômica composta
        return saida_especie, features, sigmas