// SANN.cpp: define el punto de entrada de la aplicación de consola.
//
#include "stdafx.h"
#include "SANN.h"

SANN::~SANN(){

}

/*********************************************************************************************
*	Funcion privada de la clase de clasificacion SANN
*	Descriptors1 -> Matriz original con los descriptores de esntrenamiento
*	Descriptors2 -> Matriz original con los descriptores de clasificacion
*	Matches		 -> Indices Resultado de match
*
*	TODO: 
*		  Agregar el rango dinamico
*		  Hacer que el coeficiente sea adaptativo
*********************************************************************************************/
void SANN::Match(cv::Mat &Descriptors1, cv::Mat &Descriptors2, std::vector<cv::DMatch> &Matches){
	//Primero, entrenar con el primer argumento
	if(Descriptors1.rows > 0 && Descriptors1.cols > 0)
		train(Descriptors1);
	else
		std::cout << "No se puede entrenar, la matriz 1 no contiene muestras o caracteristicas \n";

	if((Descriptors2.rows > 0 && Descriptors2.cols > 0))
		Descriptors2.copyTo(Descriptores2);
	else
		std::cout << "No se puede clasificar, la matriz 2 no contiene muestras o caracteristicas \n";

	if(Descriptors1.cols != Descriptors2.cols)
		std::cout << "No se puede clasificar, la matriz 1 y la matriz 2 no tienen el mismo numero de caracteristicas \n";

	muestrasClasificacion = Descriptors2.rows;
	
	//Distribuir aleatoriamente las muestras de clasificacion
	randomDistribution(muestrasEntrenamiento, muestrasClasificacion);

	//Calcular las distancias
	for(int i=0; i < muestrasEntrenamiento; i++){
		int indexIM = i;
		int indexIC = Material.at<float>(i,1);
		if(indexIC != -1)
			Material.at<float>(i,2) = distance(indexIM, indexIC);
	}

	//Proponer vector de cambio
	for(int i=0; i<1500; i++){
		float coef = std::exp(-i*coeficiente);
		proposeRandomPair(coef);
	}

	//Llenar la matriz de Match
	for(int i=0; i < muestrasEntrenamiento; i++)
		if(Material.at<float>(i,1) != -1 && Material.at<float>(i,2)<0.50)
			Matches.push_back(cv::DMatch(Material.at<float>(i,0), Material.at<float>(i,1), 0, Material.at<float>(i,2)));

}

/*********************************************************************************************
*	Funcion privada de la clase de clasificacion SANN
*	Descriptors -> Matriz original con los descriptores adentro
*
*********************************************************************************************/
void SANN::train(cv::Mat Descriptors){
	//Hallar el numero de muestras y el numero de caracteristicas
	muestrasEntrenamiento = Descriptors.rows;
	caracteristicas = Descriptors.cols;

	//Declarar las matrices de Sumas y desviaciones
	Sumas = cv::Mat(1,caracteristicas,CV_32FC1,cv::Scalar(0));
	Medias = cv::Mat(1,caracteristicas,CV_32FC1,cv::Scalar(0));
	Desviaciones = cv::Mat(1,caracteristicas,CV_32FC1,cv::Scalar(0));

	//Hallas las sumas
	for(int i=0; i<muestrasEntrenamiento; i++)
		for(int j=0; j<caracteristicas; j++)
			Sumas.at<float>(0,j) =  Sumas.at<float>(0,j) + Descriptors.at<float>(i,j);

	//Hallar las medias
	for(int j=0; j<caracteristicas; j++)
			Medias.at<float>(0,j) = Sumas.at<float>(0,j) / muestrasEntrenamiento;

	//Hallar las desviaciones estandar
	for(int i=0; i<muestrasEntrenamiento; i++)
		for(int j=0; j<caracteristicas; j++)
			Desviaciones.at<float>(0,j) =  Desviaciones.at<float>(0,j)  + cv::pow(Descriptors.at<float>(i,j) - Medias.at<float>(0,j), 2);
	
	for(int j=0; j<caracteristicas; j++)
		Desviaciones.at<float>(0,j) = cv::sqrt(Desviaciones.at<float>(0,j) / muestrasEntrenamiento);

	//Hallar la columna con mayor desviacion estandar
	int maxDesv = -9999999999, maxDesvIdx = 0;
	for(int i=0; i<caracteristicas; i++)
		if(Desviaciones.at<float>(0,i) > maxDesv){
			maxDesv = Desviaciones.at<float>(0,i);
			maxDesvIdx = i;
		}

	cv::Mat idx_caracteristicas;
	cv::sortIdx(Desviaciones, idx_caracteristicas, CV_SORT_EVERY_ROW + CV_SORT_ASCENDING);
	
	//Crear el material, para optimizar el proceso es una matriz con N filas como Descriptores de entrenamiento
	//y 4 columnas:
	// ) Indice de fila de la muestra material [IM]
	//0) Indice de fila de la muestra en el descriptor de entrenamiento original [IE]
	//1) Indice de fila de la muestra en el descriptor de clasificacion original [IC]
	//2) Distancia entre las muestras [D]
	Material = cv::Mat::zeros(muestrasEntrenamiento, 3, CV_32F);

	for(int i=0; i < muestrasEntrenamiento; i++){
		Material.at<float>(i,0) = -1;
		Material.at<float>(i,1) = -1;
		Material.at<float>(i,2) = 150000;
	}

	//Finalmente se ordenan los descriptores por la columna con mayor desviacion
	sortByCol(Descriptors, Descriptores1, idx_caracteristicas);
}

/*********************************************************************************************
*	Funcion privada de la clase de clasificacion SANN
*	src -> Matriz original con las muestras desordenadas
*	dst -> Matriz resultado ordenada
*	col -> Indice de la columna que se quiere ordenar
*	
*********************************************************************************************/
void SANN::sortByCol(cv::Mat &src, cv::Mat &dst, cv::Mat &col){
	//Primero se crea la matriz de destino igual a la matriz fuente y una temporal
	cv::Mat sorted = cv::Mat::zeros(src.rows,src.cols,src.type());
	cv::Mat tmp = cv::Mat::zeros(src.rows,src.cols+1,src.type());
	cv::Mat tmp_sorted = cv::Mat::zeros(src.rows,src.cols+1,src.type());

	//Segundo se crea la matriz de indice original y se concatena a la original
	cv::Mat originalIndex = cv::Mat(src.rows, 1, src.type());
	for(int i=0; i<src.rows; i++)
		originalIndex.at<float>(i,0) = i;
	cv::hconcat(src, originalIndex, tmp);

	//Ordenar para cada columna
	for(int i=0; i<col.cols; i++){
		//Determinar cual es la columna que toca ordenar
		int orderCol = col.at<int>(0,i);

		//Despues hallar el orden de los indices de cada columna
		cv::Mat idx;
		cv::sortIdx(tmp, idx, CV_SORT_EVERY_COLUMN + CV_SORT_ASCENDING);

		//Se itera y se copia fila por fila
		for(int k=0; k<sorted.rows; k++){
			int indiceCopiar = idx.at<int>(k,orderCol);
			tmp.row(indiceCopiar).copyTo(tmp_sorted.row(k));
		}
		//Finalmente se pasa de tmp_sorted a tmp
		tmp_sorted.copyTo(tmp);
	}
	for(int i=0; i<src.rows; i++)
		Material.at<float>(i,0) = tmp.at<float>(i,64);

	//Se copia a la salida
	tmp(cvRect(0,0,64,tmp.rows)).copyTo(dst);
}

/*********************************************************************************************
*	Funcion privada de la clase de clasificacion SANN
*	N -> Numero de  muestras de entrenamiento
*	M -> Numero de muestras a clasificar
*
*	!IMPORTANTE:  N => M
*********************************************************************************************/
void SANN::randomDistribution(int N, int M){
	//Crear el vector base y llenarlo
	std::vector<int> listaBaseEntrenamiento;
	for(int i=0; i < M; i++)
		listaBaseEntrenamiento.push_back(i);

	//Crear la lista desordenada y llenarla
	std::vector<int> listaBaseClasificacion;
	srand (time(NULL));
	int El = M;
	for(int i=0; i < M; i++){
		int index = rand() % El;
		listaBaseClasificacion.push_back(listaBaseEntrenamiento.at(index));
		listaBaseEntrenamiento.erase(listaBaseEntrenamiento.begin() + index);
		if(!--El)
			break;
	}

	//Crear la lista de pareja iniciarla en -1
	std::vector<int> listaBasePareja;
	for(int i=0; i < N; i++)
		listaBasePareja.push_back(-1);

	//Ubicar los indice de la lista de baseClasificacion en la de pareja
	El = M;
	for(int i=0; i < M; i++){
		while(true){
			int index = rand() % N;
			if(listaBasePareja.at(index) == -1){
				listaBasePareja.at(index) = listaBaseClasificacion.at(i);
				break;
			}
		}
	}

	//Finalmente llenar los indices en el Material
	for(int i=0; i < N; i++)
		Material.at<float>(i,1) = listaBasePareja.at(i);
}

/*********************************************************************************************
*	Funcion privada de la clase de clasificacion SANN
*	N -> Indice de la muestra de entrenamiento
*	M -> Indice de la muestra de clasificacion
*
*	IMPORTANTE: N,M deben ser valores existentes!
*	TODO: Optimizar con absdiff().sum()
*********************************************************************************************/
float SANN::distance(int N, int M){
	float d = 0;
	for(int i=0; i<caracteristicas; i++){
		float  resta = Descriptores1.at<float>(N,i) - Descriptores2.at<float>(M,i);
		d += std::abs(resta);
	}

	return d;
}

/*********************************************************************************************
*	Funcion privada de la clase de clasificacion SANN y metodo principal basado en SA
*
*	TODO: Optimizar con absdiff().sum()
*		
*********************************************************************************************/
void SANN::proposeRandomPair(float coeff){
	int Ncambios = 0;
	int NcambiosAleatorios = 0;
	//Recorrer el material en busca de particulas a clasificar
	srand (time(NULL));
	int El = muestrasEntrenamiento;
	for(int i=0; i < muestrasEntrenamiento; i++){
		int indexIC = Material.at<float>(i,1);
		//Si encuentra una particula de clasificacion
		if(indexIC != -1){
			//Buscar en el espacio de indices propuesto una posibilidad de cambio
			int IMAleatorio = rand() % El;
			//Se calcula la funcion de costo
			float d_proposed = distance(IMAleatorio,indexIC);
			float d_actual = Material.at<float>(i,2);

			//Se calcula un numero aleatorio entre 0 y 1
			float aleatorio_unitario = (rand() % 100)/100.0;

			//Si se mejora la funcion de costo o la funcion de probabilidad la acepta
			if((d_proposed < d_actual || aleatorio_unitario < coeff) && i != IMAleatorio){
				
				//Para hacer debug, imprimir primero los descriptores propuestos
				if(doDebug){
					std::cout << "\n \n Material antes de: \n";
					DEBUG(i,IMAleatorio,false);
				}			

				if(aleatorio_unitario < coeff) NcambiosAleatorios++;
				//Si el espacio esta vacio, entonces se pasa la particula a ese espacio
				if(Material.at<float>(IMAleatorio,1) == -1){
					Material.at<float>(IMAleatorio,1) = indexIC;
					Material.at<float>(IMAleatorio,2) = d_proposed;
					Material.at<float>(i,1) = -1;
					Material.at<float>(i,2) = 150000;
					Ncambios++;
				}
				//Sino esta vacio, se verifica que la funcion de costo mejore respecto al valor actual
				else if(d_proposed < Material.at<float>(IMAleatorio,2) || aleatorio_unitario < coeff){
					Material.at<float>(i,1) = Material.at<float>(IMAleatorio,1);
					Material.at<float>(i,2) = distance(i, Material.at<float>(i,1));					
					Material.at<float>(IMAleatorio,1) = indexIC;
					Material.at<float>(IMAleatorio,2) = d_proposed;
					Ncambios++;
				}

				//Si se debe hacer debug se imprime el material despues de, y los descriptores afectados
				if(doDebug){
					std::cout << "\n Material despues de: \n";
					DEBUG(i,IMAleatorio,true);
				}
			}
		}
	}

	//std::cout << Ncambios << " " << NcambiosAleatorios << " " << distanciaPromedio() << " " << coeff <<"\n";
}

/*********************************************************************************************
*	Funcion privada de la clase de clasificacion SANN que imprime el material
*
*********************************************************************************************/
void SANN::toString(){
	std::cout << "Material: \n \n";
	for(int i=0; i < muestrasEntrenamiento; i++){
		for(int j=0; j < 3; j++)
			std::cout << Material.at<float>(i,j) << " ";
		std::cout << "\n";
	}
}

/*********************************************************************************************
*	Funcion privada de la clase de clasificacion SANN que imprime el descriptor1 en la muestra
*	solicitada
*	
*	row	->	Fila solicitada
*
*********************************************************************************************/
void SANN::descriptor1AtRow(int row){
	if(row >= 0){
		for(int i=0; i<Descriptores1.cols; i++)
			std::cout << Descriptores1.at<float>(row,i) << " ";
		std::cout << "\n";
	}
}

/*********************************************************************************************
*	Funcion privada de la clase de clasificacion SANN que imprime el descriptor2 en la muestra
*	solicitada
*
*	row	->	Fila solicitada
*
*********************************************************************************************/
void SANN::descriptor2AtRow(int row){
	if(row >= 0){
		for(int i=0; i<Descriptores2.cols; i++)
			std::cout << Descriptores2.at<float>(row,i) << " ";
		std::cout << "\n";
	}
}

/*********************************************************************************************
*	Funcion privada de la clase de clasificacion SANN que imprime el rastro DEBUG de los cambios
*	realizados durante la clasifcacion
*
*	m1	->	Primer indice del material
*	m2	->	Segundo indice del material
*	print_descriptors	->	Si es true, se imprimen los descriptores de las filas
*
*********************************************************************************************/
void SANN::DEBUG(int m1, int m2, bool print_descriptors){

	std::cout << Material.at<float>(m1,0) << " "<< Material.at<float>(m1,1) << " "<< Material.at<float>(m1,2) <<"\n";
	std::cout << Material.at<float>(m2,0) << " "<< Material.at<float>(m2,1) << " "<< Material.at<float>(m2,2) <<"\n";

	if(print_descriptors){
		std::cout << "\n Descriptores usados: \n";
		descriptor1AtRow(m1);
		descriptor2AtRow(Material.at<float>(m1,1));
		descriptor1AtRow(m2);
		descriptor2AtRow(Material.at<float>(m2,1));
	}

}

/*********************************************************************************************
*	Funcion privada de la clase de clasificacion SANN que calcula la distancia promedio actual
*	de las particulas con pareja
*
*********************************************************************************************/
float SANN::distanciaPromedio(){
	float total = 0;
	
	for(int i=0; i < muestrasEntrenamiento; i++)
		if(Material.at<float>(i,2) != 150000)
			total += Material.at<float>(i,2);

	return total/muestrasEntrenamiento;
}

/*********************************************************************************************
*	Funcion publica de la clase de clasificacion SANN que establece el coeficiente para el 
*	calculo de la temperatura
*
*********************************************************************************************/
void SANN::setCoefficiente(float coeff){
	coeficiente = coeff;
}