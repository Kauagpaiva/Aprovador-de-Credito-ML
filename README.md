# Informações sobre o Classificador:
> Neste trabalho eu tive que construir um classificador para apoio à decisão de aprovação de crédito.

A ideia foi identificar, dentre os clientes que solicitam um produto de crédito (como um cartão de crédito ou um empréstimo pessoal, por exemplo) e que cumprem os pré-requisitos essenciais para a aprovação do crédito, aqueles que apresentem alto risco de não conseguirem honrar o pagamento, tornando-se inadimplentes.

Para isso, eu recebi um arquivo com dados históricos de 20.000 solicitações de produtos de créditos que foram aprovadas pela instituição, acompanhadas do respectivo desfecho, ou seja, acompanhadas da indicação de quais desses solicitantes conseguiram honrar os pagamentos e quais ficaram inadimplentes.

Com base nesses dados históricos, eu construi um classificador que, a partir dos dados de uma nova solicitação de crédito, tenta predizer se este solicitante será um bom ou mau pagador.

## Tecnologias utilizadas
O projeto foi desenvolvido utilizando as seguintes tecnologias:

- Python
- Scikit-Learn
