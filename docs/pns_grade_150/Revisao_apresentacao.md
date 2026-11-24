# Revisão da apresentação da PNS 2019

Aplicar em uma cópia editável da apresentação de 21 páginas. Usar a tipografia, a paleta e a estrutura dos slides existentes. Manter a sequência de apresentadores de cada bloco.

| Slide | Responsável no roteiro | Alteração |
|---|---|---|
| 2 | Pietro | Trocar a observação sobre o 11º preditor por “Dez preditores sociodemográficos e um alvo binário”. |
| 7–8 | Rafael | Apresentar o kNN de referência com UF original, RobustScaler e one-hot. AUC-ROC média de 0,6281 e F1 médio de 0,3590 nos dez folds. A matriz exibida é acumulada fora do treino. Caso mantenham a precisão, usar 44,51% para essa matriz. |
| 9 | Pedro e Pietro | Substituir a pendência por “150 receitas avaliadas · dez folds por receita”. Mostrar 40 históricas e 110 novas, com kNN padrão e teste reservado. Mover mudanças de limiar e pesos de votação para o roteiro dos diagnósticos. |
| 10–13 | Bloco de experimentos | Manter a identificação dos diagnósticos. Nos slides de matrizes acumuladas, rotular AUC e F1 agregados como OOF global. |
| 14 | Bloco de experimentos | Manter o título “O que cada estratégia entregou?” e o subtítulo “Média dos dez folds”. H15, H17 e H19 também constam da grade consolidada. Os ensaios TSTR aparecem somente como complementares. |
| Após o 14 | Bloco de experimentos | Inserir o slide “As melhores receitas da grade”, com o gráfico de dispersão e os três destaques abaixo. |
| 17 e 19 | José Arthur e bloco de conclusão | Limitar a afirmação sobre proximidade aos pares analisados na janela de minutos. Tratar sobreposição como hipótese e preservar a identificação da figura simulada do slide 20. |
| 21 | José Arthur | Atualizar a conclusão para distinguir as melhores receitas por AUC e F1, registrar a preservação do teste e encerrar com a magnitude observada dos ganhos. |

## Slide da grade consolidada

**As melhores receitas da grade**

150 combinações · 1.500 avaliações · kNN padrão

| Destaque | Receita | AUC-ROC média | F1 médio |
|---|---|---:|---:|
| Maior AUC da grade | H17 · Cinco regiões e Tomek Links | 0,6380 | 0,4191 |
| Maior F1 da grade | H19 · Cinco regiões, SMOTENC e Tomek Links | 0,6228 | 0,4514 |
| Maior AUC entre as novas | G098 · Z-score, IQR, seleção e PCA | 0,6349 | 0,3716 |

O gráfico `auc_f1_grade.png` mostra um ponto por configuração. Se a tabela e o gráfico disputarem espaço, usar o gráfico e destacar três valores em cartões. Os demais números permanecem nas notas de fala.

## Notas de fala

**Pedro e Pietro, ao apresentar a grade.** “Consolidamos 150 combinações de pré-processamento, cada uma avaliada em dez folds. A maior AUC foi 0,6380 com cinco regiões e Tomek. SMOTENC com Tomek obteve o maior F1, com mais falsos positivos. Entre as novas combinações, destaque para a G098, com AUC de 0,6349.”

**José Arthur, ao interpretar os resultados.** “Em relação à referência regional, o Tomek identificou, por fold, cerca de 202 positivos a mais e produziu cerca de 321 falsos positivos adicionais. É um ganho com custo mensurável. Os resultados são da validação no desenvolvimento. O teste ficou reservado.”

**Conclusão em tela.** “Tomek liderou em AUC. SMOTENC + Tomek liderou em F1. Os ganhos de AUC foram modestos.”

Após inserir a página, atualizar os números do rodapé e conferir os rótulos OOF global e média dos folds.
