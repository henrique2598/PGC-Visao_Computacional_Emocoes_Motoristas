#-----------------------------------------------------------------------------
#-                    Projeto de Graduação em Computação                     -
#-                            UFABC - Santo André                            -
#-                                                                           -
#-  Nome: Henrique Vicente Ferraro Oliveira                                  -
#-  Ra: 11201721650                                                          -
#-  Orientador: Prof. Dr. Alexandre Noma                                     -
#- Título: Uso de visão computacional para detecção de emoções de motoristas -
#-----------------------------------------------------------------------------



#-------------------
#-   Importações   -
#-------------------
#Python 3.10.0
import json
import time
#pip install numpy==1.26.4
import numpy as np
#pip install opencv-python
import cv2
#pip install tensorflow==2.9.1
from keras.utils import img_to_array
from keras.models import model_from_json



#--------------------------------------
#-   Definição dos paths e variáveis  -
#--------------------------------------
Camera_Index = 0
winName = 'PGC - Uso de visão computacional para detecção de emoções de motoristas'
Path_TelaInicial = "Assets/Images/Background-Inicial.png"
Path_TelaCorrida = "Assets/Images/Background-Corrida.png"
Path_TelaFinal = "Assets/Images/Background-Final.png"
Path_JsonDriverDetails = "Assets/Driver/driver_details.json"
Path_CascadeClassifier = "Assets/Models/haarcascade_frontalface_default.xml"
Path_FacialModel = "Assets/Models/facial_expression_model_structure.json"
Path_FacialWeights = "Assets/Models/facial_expression_model_weights.h5"



#----------------------
#-   Inicializações   -
#----------------------
# Definição do classificador para detecção dos rostos
face_cascade = cv2.CascadeClassifier(Path_CascadeClassifier)

# Definiçção do modelo utilizado para reconhecimento da expressão facial
model = model_from_json(open(Path_FacialModel, "r").read())

# Definição dos pesos do modelo utilizado para reconhecimento da expressão facial
model.load_weights(Path_FacialWeights)

# Definição das emoções suportadas
emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

# Definição da entrada de vídeo
cap = cv2.VideoCapture(Camera_Index)

# Criação do contador de emoções
CounterEmotions = {'angry': 0, 'disgust': 0, 'fear': 0, 'happy': 0, 'sad': 0, 'surprise': 0, 'neutral': 0}

# Carrega o backgroud de cada etapa
TelaInicial = cv2.imread(Path_TelaInicial)
TelaCorrida = cv2.imread(Path_TelaCorrida)
TelaFinal = cv2.imread(Path_TelaFinal)

# Define a janela de exibição das imagens
cv2.namedWindow(winName, cv2.WINDOW_FULLSCREEN)

# Posiciona a janela na posição (50, 50)
cv2.moveWindow(winName, 50, 50)

# Leitura do json com as informações do motorista
with open(Path_JsonDriverDetails, "r", encoding="utf-8") as j:
     driver_details = json.loads(j.read())



#--------------------------------------------------------
#-   Função para identificação da emoção do motorista   -
#--------------------------------------------------------
def DetectaEmocao(detected_face):
	# Converte a face para um array
	img_pixels = img_to_array(detected_face)
	img_pixels = np.expand_dims(img_pixels, axis = 0)

	# Normalização do aary de pixels, escala [0, 255] para escala [0, 1]
	img_pixels /= 255 

	# Função responsável pela predição da emoção com base no modelo
	predictions = model.predict(img_pixels) #store probabilities of 7 expressions

	# Detecta a emoção que mais se assemelha
	max_index = np.argmax(predictions[0])
	emotion = emotions[max_index]

	# Retorna a emoção
	return emotion



#----------------------------------------------
#-   Função para calcular a nota da corrida   -
#----------------------------------------------
def CalculaNota(CounterEmotions, DuracaoCorrida, CounterTimes):
	# Contabiliza o total de emoções detectadas
	totalEmotions = sum(CounterEmotions.values())
	# Contabiliza o total de emoções negativas detectadas
	negativeEmotions = CounterEmotions['angry']+CounterEmotions['disgust']+CounterEmotions['fear']+CounterEmotions['sad']
	# Contabiliza o total de emoções positivas detectadas
	positiveEmotions = CounterEmotions['happy']
	# Contabiliza a proporção de emoções de interesse em relação ao total
	percentEmotions = float(negativeEmotions-positiveEmotions)/float(totalEmotions)

	# Vetor para armazenar os step times
	CounterStepTimes=[]
	# Percorre os tempos entre as detecções
	for i in range(len(CounterTimes)-1):
		# Calcula o step time
		StepTime = CounterTimes[i+1]-CounterTimes[i]
		# Adiciona o step time ao vetor, desde que ele seja menor que 3 segundos
		if (StepTime<3):
			CounterStepTimes.append(StepTime)
	# Calcula o step time médio
	emotionsStepTime = sum(CounterStepTimes) / len(CounterStepTimes)
	# Calcula o tempo de cobertura pela detecção
	emotionsTime = totalEmotions * emotionsStepTime
	# Calcula a porcentagem da corrida que foi coberta pela detecção
	percentCobertura = emotionsTime/float(DuracaoCorrida)

	# Calcula a nota final
	nota = 5.00 - (5.00 * (percentEmotions * percentCobertura))
	# Valida a nota final no intervalo [0, 5]
	if(nota<0):
		nota = 0.00
	elif(nota>5):
		nota=5.00
	
	# Retorna a nota arredondada com 2 casas decimais
	return round(nota, 2)



#-------------------------------------
#-   Atualizar json driver details   -
#-------------------------------------
def AtualizaJson(Nota_corrida):
	# Abre o arquivo JSON
	with open(Path_JsonDriverDetails, "r") as jsonFile:
		data = json.load(jsonFile)

	# Acrescenta mais uma viagem
	data["Viagens"] += 1
	# Acrescenta a nota da corrida ao histórico
	data["HistoricoDeNotas"].append(Nota_corrida)
	# Atualiza a nota de comportamento atual com  a média do histórico
	NotaAtualizada = np.mean(data["HistoricoDeNotas"])
	data["Nota comportamento"] = round(NotaAtualizada, 2)

	# Salva o arquivo JSON
	with open(Path_JsonDriverDetails, "w") as jsonFile:
		json.dump(data, jsonFile)



#----------------------------
#-      Loop Principal      -
#----------------------------

# Variáveis de estado do programa
EmExpediente=True
EmCorrida=False
ResultadosCorrida=True

# Loop principal
while(EmExpediente==True):
	# Leitura do JSON com as informações da turma
	with open(Path_JsonDriverDetails, "r", encoding="utf-8") as j:
		driver_details = json.loads(j.read())
	# Leitura do background da tela inicial
	TelaInicial = cv2.imread(Path_TelaInicial)
	# Adiciona as informações do motorista na tela inicial
	cv2.putText(TelaInicial, driver_details['Motorista'], (170, 325), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)
	cv2.putText(TelaInicial, driver_details['Veiculo'], (135, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)
	cv2.putText(TelaInicial, str(driver_details['Viagens']), (140, 395), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)
	cv2.putText(TelaInicial, str(driver_details['Nota passageiros']), (260, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)
	cv2.putText(TelaInicial, str(driver_details['Nota comportamento']), (320, 465), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)
	#Exibe tela inicial com informações do motorista
	cv2.imshow(winName, TelaInicial)

	# Decide se vai iniciar uma nova corrida ou encerrar o dia
	while (True):
		# Iniciar uma nova corrida
		if cv2.waitKey(1) & 0xFF == ord('i'):
			EmCorrida = True
			CounterEmotions = {'angry': 0, 'disgust': 0, 'fear': 0, 'happy': 0, 'sad': 0, 'surprise': 0, 'neutral': 0}
			CounterTimes = []
			Total_emotions = 0
			start_time = time.time()
			break
		# Encerrar o dia
		elif cv2.waitKey(1) & 0xFF == ord('f'):
			EmExpediente = False
			break

	# Laço de repetição durante a corrida - Captura de emoções
	while(EmCorrida==True):
		# Lê a webcam
		ret, cam = cap.read()
		cam = cv2.resize(cam, (405,303), interpolation=cv2.INTER_AREA)
		gray = cv2.cvtColor(cam, cv2.COLOR_BGR2GRAY)

		# Detecta os rostos
		faces = face_cascade.detectMultiScale(gray, 1.3, 5)
		
		# Limpa as variáveis
		Driver_emotion = ""

		# Carrega o background da tela de corrida
		TelaCorrida = cv2.imread(Path_TelaCorrida)

		# Percorre os rostos detectados
		for (x,y,w,h) in faces:
			cv2.rectangle(cam,(x,y),(x+w,y+h),(255,0,0),2) # Desenha o retângulo ao redor do rosto

			# Recorta o rosto detectado e pré-processa a imagem
			detected_face = cam[int(y):int(y+h), int(x):int(x+w)]  # Recorte do rosto
			detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY) # Transforma para escala de cinza
			detected_face = cv2.resize(detected_face, (48, 48)) # Redimensiona para 48x48

			# Função que detecta a emoção
			Driver_emotion = DetectaEmocao(detected_face)
			# Adiciona o tempo para cálculo de step time
			CounterTimes.append(time.time())
			# Atualiza o contador de emoções
			CounterEmotions[Driver_emotion] += 1
			# Atualiza a contagem de emoções
			Total_emotions += 1


		# Adiciona as informações atualizadas na tela
		cv2.putText(TelaCorrida, str(Driver_emotion), (100, 125), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
		cv2.putText(TelaCorrida, str(Total_emotions), (100, 225), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)

		# Concatena o placar com o vídeo
		img = cv2.vconcat([cam, TelaCorrida])

		# Exibe a tela
		cv2.imshow(winName,img)

		# Verifica se o cadastro foi concluído e atualiza variáveis
		if cv2.waitKey(1) & 0xFF == ord('q'):
			EmCorrida = False
			ResultadosCorrida = True
			end_time = time.time()

	# Exibe as métricas da corrida finalizada
	while(ResultadosCorrida==True):
		# Carrega backgrounda da tela final e exibe as informações de resultado
		TelaFinal = cv2.imread(Path_TelaFinal)
		DuracaoCorrida = str('{0:.2f}'.format(end_time - start_time))
		cv2.putText(TelaFinal, driver_details['Motorista'], (165, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)
		cv2.putText(TelaFinal, "UFABC SA", (100, 177), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)
		cv2.putText(TelaFinal, "UFABC SBC", (125, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)
		cv2.putText(TelaFinal, DuracaoCorrida, (135, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)

		# Laço para exibir os resultados por emoção
		y0, dy = 350, 30
		chaves = list(CounterEmotions.keys())
		valores = list(CounterEmotions.values())
		for i in range (len(chaves)):
			y = y0 + i*dy
			line = str(chaves[i])+": "+str(valores[i])
			cv2.putText(TelaFinal, line, (50, y ), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)

		# Exibe nota da corrida
		Nota_corrida = CalculaNota(CounterEmotions, DuracaoCorrida, CounterTimes)
		cv2.putText(TelaFinal, str(Nota_corrida), (295, 587), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)
		cv2.imshow(winName,TelaFinal)

		# Valida a mudança de tela por parte do usuário
		while(True):
			# Verifica se o motorista deseja voltar ao menu inicial
			if cv2.waitKey(1) & 0xFF == ord('i'):
				AtualizaJson(Nota_corrida)
				ResultadosCorrida=False
				break



# Fechas a câmera e janela aberta
cap.release()
cv2.destroyAllWindows()