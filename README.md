# Efeito Fotoelétrico
O objetivo desse programa é remover os ruídos dos dados do Experimento 1 de Física Experimental V.  
O programa pode ser encontrado clicando [aqui](https://github.com/MatheusJCastro/Code_Lab_V/) ou em:

> <https://github.com/MatheusJCastro/Code_Lab_V/>

## Formato dos arquivos
Os dados devem estar no formato: `DDMM_COR_INTENSIDADE.csv`  
Exemplo: `0503_amarelo_100.csv`  
O programa foi feito para funcionar com o arquivo original gerado pelo ProKeithley 6487.  
Todos os arquivos devem estar na mesma pasta em que o *medidas.py*.  

## Execução
Para rodar o código é necessario ter *Python3*, *Numpy* e *Matplotlib* instalados.  
Em sistemas linux rode o comando no terminal para executar o programa:
	
	python3 medidas.py

## Configuração
**Para configurar o programa abra o arquivo "medidas.py" e modifique as seguintes linhas:**
	
* Para colocar os dias em que houve tomada de dados:
		
		dias = [...]
	
	Os dias devem estar no formado DDMM, entre aspas duplas (") e separados por vírgulas. Exemplo:
	
		dias = ["0503", "0603"]
	
* Modifique o *1* em *bias=1* e *lamp=1* para *0* caso não queria tirar algum dos tipos de ruído.
	
		reduced = noise_removal(dt, noi_bias, noi_lamp, bias=1, lamp=1)
		
### Salvar novos dados em *.csv*

* Para salvar todos os dados sem o ruído no formato csv, descomente a linha:

		save_new_data(reduced)
		
	O programa criará uma pasta com o nome `reduced_data` e salvará todos os arquivo individualmente no mesmo formato dos arquivos de entrada, acrescentando apenas *"reduced"* no final de cada arquivo.  
	**Atenção:** Cada vez que executado, o programa sobrescreve os arquivos que contêm o mesmo nome.

### Plotar e Salvar Gráficos

* Descomente essa linha caso queira plotar todos os gráficos de frequência:

		plot_all(reduced, save=False)

	Troque False para True caso queira salvar uma imagem png do gráfico.
	
* Descomente essa linha caso queira plotar um gráfico de uma intensidade k (todas frequências):
		
		plot_same_intensity(reduced, save=False, k=100)

	Troque False para True caso queira salvar uma imagem png do gráfico.
