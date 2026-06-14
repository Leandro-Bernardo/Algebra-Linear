import torch
import torch.nn as nn

class HierarchicalTaxonomicLoss(nn.Module):
    def __init__(self, lambda_tax: float = 0.2, k: int = 6, eps: float = 1e-7):
        """
        Args:
            lambda_tax: Peso do regularizador taxonómico (λ).
            k: Número de níveis taxonómicos (padrão: 6).
            eps: Pequena constante para estabilidade numérica contra NaN.
        """
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.lambda_tax = lambda_tax
        self.k = k
        self.eps = eps

    def forward(self, saida_especie, classe_alvo, sigmas, codigos_taxonomicos):
        """
        Args:
            saida_especie: [Batch, num_classes] -> Saída linear do modelo.
            classe_alvo: [Batch] -> IDs inteiros das espécies reais.
            sigmas: [Batch, k] -> Valores singulares extraídos pelo SVD no modelo.
            codigos_taxonomicos: [Batch, k] -> Matriz com os IDs de todos os níveis de cada amostra.
        """
        ## CLASSIFICAÇÃO TRADICIONAL
        l_ce = self.ce_loss(saida_especie, classe_alvo)

        ## REGULARIZADOR TAXONÓMICO (Vetorizado na GPU)
        # Expande os códigos taxonomicos para comparar todas as amostras contra todas (All-to-All)
        codigos_i = codigos_taxonomicos.unsqueeze(1) # [Batch, 1, k]
        codigos_j = codigos_taxonomicos.unsqueeze(0) # [1, Batch, k]

        # Conta quantos níveis são idênticos entre a amostra i e j
        niveis_iguais = (codigos_i == codigos_j).sum(dim=-1).float() # [Batch, Batch]

        # Distância taxonómica alvo: 1.0 se não partilharem nada, 0.0 se forem identidades
        distancia_alvo = 1.0 - (niveis_iguais / self.k) # [Batch, Batch]

        # CORREÇÃO DE ESTABILIDADE NUMÉRICA
        sigmas_i = sigmas.unsqueeze(1) # [Batch, 1, k]
        sigmas_j = sigmas.unsqueeze(0) # [1, Batch, k]

        # Quadrado das diferenças
        delta_ao_quadrado = torch.sum((sigmas_i - sigmas_j) ** 2, dim=-1)

        # Adiciona um fator de estabilidade EPS na raiz quadrada para evitar que o gradiente exploda para Infinito quando a distância entre sigmas for zero
        distancia_predita = torch.sqrt(delta_ao_quadrado + self.eps) # [Batch, Batch]

        # Cria uma máscara para ignorar a diagonal principal (comparação da imagem com ela mesma)
        mascara = ~torch.eye(sigmas.size(0), dtype=torch.bool, device=sigmas.device)

        # Calcula o erro quadrático médio apenas nos pares válidos da matriz
        erro_distancia = (distancia_predita - distancia_alvo) ** 2
        l_tax = torch.mean(erro_distancia[mascara])

        ## 3. SOMA COMPOSTA
        loss_total = l_ce + (self.lambda_tax * l_tax)

        return loss_total, l_ce, l_tax

# class HierarchicalTaxonomicLoss(nn.Module):
#     def __init__(self, lambda_tax: float = 0.2, k: int = 6, eps: float = 1e-7):
#         """
#         Args:
#             lambda_tax: Peso do regularizador taxonómico (λ).
#             k: Número de níveis taxonómicos (padrão: 6).
#         """
#         super().__init__()
#         self.ce_loss = nn.CrossEntropyLoss()
#         self.lambda_tax = lambda_tax
#         self.k = k
#         self.eps = eps

#     def forward(self, saida_especie, classe_alvo, sigmas, codigos_taxonomicos):
#         """
#         Args:
#             saida_especie: [Batch, num_classes] -> Saída linear do modelo.
#             classe_alvo: [Batch] -> IDs inteiros das espécies reais.
#             sigmas: [Batch, k] -> Valores singulares extraídos pelo SVD no modelo.
#             codigos_taxonomicos: [Batch, k] -> Matriz com os IDs de todos os níveis de cada amostra.
#         """
#         ## CLASSIFICAÇÃO TRADICIONAL
#         l_ce = self.ce_loss(saida_especie, classe_alvo)

#         ## REGULARIZADOR TAXONÓMICO
#         # Estratégia Vetorizada Barata (Broadcasting na GPU)
#         # Expandimos os códigos para comparar todas as amostras contra todas (All-to-All)
#         # codigos_tax: [Batch, k] -> codigos_i: [Batch, 1, k] e codigos_j: [1, Batch, k]
#         codigos_i = codigos_taxonomicos.unsqueeze(1)
#         codigos_j = codigos_taxonomicos.unsqueeze(0)

#         # Conta quantos níveis são idênticos entre a amostra i e j
#         niveis_iguais = (codigos_i == codigos_j).sum(dim=-1).float() # [Batch, Batch]

#         # Distância taxonómica alvo: 1.0 se não partilharem nada, 0.0 se forem a mesma espécie
#         distancia_alvo = 1.0 - (niveis_iguais / self.k) # [Batch, Batch]

#         # Agora calculamos a distância geométrica predita no espaço latente usando os Sigmas
#         # sigmas: [Batch, k] -> sigmas_i: [Batch, 1, k] e sigmas_j: [1, Batch, k]
#         sigmas_i = sigmas.unsqueeze(1)
#         sigmas_j = sigmas.unsqueeze(0)

#         # Distância Euclidiana entre os vetores de valores singulares de cada par do lote
#         distancia_predita = torch.cdist(sigmas, sigmas, p=2.0) # [Batch, Batch]

#         # Minimizamos o Erro Quadrático Médio (MSE) entre as distâncias preditas e as alvos
#         # Usamos o MSE para penalizar severamente grandes distorções na hierarquia
#         l_tax = torch.mean((distancia_predita - distancia_alvo) ** 2)

#         # --- 3. SOMA COMPOSTA ---
#         loss_total = l_ce + (self.lambda_tax * l_tax)

#         return loss_total, l_ce, l_tax