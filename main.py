from fastapi import FastAPI
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer #para convertir las caracteristicas combinadas en una matriz
from sklearn.neighbors import NearestNeighbors

#Creamos una instancia de la clase FastAPI
app = FastAPI(
    title='Proyecto Individual 1 - MLOps',   
    description='API de películas y recomendaciones basado en ML')

#Importamos los datos
movies = pd.read_csv('datasets/movies.csv', sep=';')
moviesML = pd.read_csv('datasets/moviesML.csv')

@app.get('/')
async def root():
    return {'Proyecto Individual 1 - API de películas y recomendaciones basado en ML'}

#Retorna la cantidad de peliculas por un mes en especifico
@app.get('/cantidad_filmaciones_mes/{mes}')
def cantidad_filmaciones_mes(mes):
    mes = mes.lower()
    cantidad = len(movies[movies['release_month'] == mes])
    if(cantidad == 0):
        return {'No existen peliculas para el mes especificado!'}
    else:
        return {'mes':mes, 
                'cantidad':cantidad
                }

#Retorna la cantidad de peliculas por un dia en especifico
@app.get('/cantidad_filmaciones_dia/{dia}')
def cantidad_filmaciones_dia(dia):
    dia = dia.lower()

    numMovies = len(movies[movies['release_day'] == dia])

    if(numMovies == 0):
        return {'No existen peliculas para el dia especificado!'}
    else:
        return {'dia':dia, 
                'cantidad':numMovies
                }

#Retorna el titulo, anio y popularidad de una pelicula especifica
@app.get('/score_titulo/{titulo}')
def score_titulo(titulo):

    title = titulo.lower()
    movie = movies.query('title == @title')
    
    if title not in movie.values:
        return 'Película no encontrada! Prueba con: American Pie, Cars, The Matrix, Gladiator, Hotel Transylvania, Toy Story 3'
    else:
        anio = movie['release_year'].values[0]
        popularity = movie['popularity'].values[0]

        return {'titulo': title, 
                'anio': int(anio), 
                'popularidad': popularity
                }

#Retorna el titulo, anio, voto total y voto promedio de una pelicula
#con votos mayores a 2000
@app.get("/votos_titulo/{titulo}")
def votos_titulo(titulo):

    title = titulo.lower()
    movie = movies.query('title == @title')

    if title not in movie.values:
        return 'Película no encontrada! Prueba con: American Pie, Cars, The Matrix, Gladiator, Hotel Transylvania, Toy Story 3'
    else:
        anio = movie['release_year'].values[0]
        vote_count = movie['vote_count'].values[0]
        vote_average = movie['vote_average'].values[0]

        return {'titulo': title, 
                'anio': int(anio), 
                'voto_total': int(vote_count),
                'voto_promedio': vote_average
                }

#Retorna el nombre del actor, cantidad de peliculas, retorno total
#y retorno promedio
@app.get("/get_actor/{nombre_actor}")    
def get_actor(nombre_actor):

    actor = nombre_actor.lower()
    actor= [actor]
    movie = movies[movies['actor'].apply(lambda x: any(actor in x for actor in actor))]
    
    if movie.empty:
        return 'Actor no encontrado! Prueba con: Robert de Niro,  Winona Ryder, Johnny Depp, Denzel Washington, Julia Roberts, Mara wilson, '
    else: 
        cantidad_peliculas = int(movie.title.value_counts().sum())
        retorno_total = round(float(movie['return'].values.sum()), 2)
        promedio_retorno = round(retorno_total/cantidad_peliculas, 2)

        return {'actor': nombre_actor, 
                'cantidad_filmaciones': cantidad_peliculas, 
                'retorno_total': retorno_total,
                'retorno_promedio': promedio_retorno
                }

#Retorna el nombre del director, retorno total del director, peliculas
#del director (titulo,anio,retorno, budget y revenue de cada pelicula)
@app.get("/get_director/{nombre_director}")
def get_director(nombre_director):
    director=nombre_director.lower()
    moviesFilter = movies.query("director == @director")

    if director not in moviesFilter.values:
        return 'Director no encontrado! Prueba con: George Lucas, Robert Zemeckis, James Cameron, Steven Spielberg, Stephen Hopkins, Alfred Hitchcock'
    else:
        moviesList=[]
        
        for i in range(moviesFilter.shape[0]):
            movie={
                    "titulo":moviesFilter.title.to_list()[i],
                    "anio":moviesFilter.release_year.to_list()[i],
                    "retorno_pelicula":round(moviesFilter['return'],2).to_list()[i],
                    "budget":moviesFilter.budget.to_list()[i],
                    "revenue_pelicula":moviesFilter.revenue.to_list()[i]
                }
            
            moviesList.append(movie)

        return {'director':director,
                'retorno_total_director':float(round(moviesFilter['return'].sum(),2)),
                "peliculas":moviesList
                }

#Machine Learning
count_vectorizer  = CountVectorizer()
sparse_matrix  = count_vectorizer.fit_transform(moviesML['tags'])
# Creamos un modelo para encontrar los vecinos mas cercanos en un espacio de caracterisicaa
nn = NearestNeighbors(metric='cosine', algorithm='brute')
nn.fit(sparse_matrix)
indicesML = pd.Series(moviesML.index, index=moviesML['title']).drop_duplicates()

#Sistema de Recomendacion. Retorna las 5 peliculas mas parecidas
@app.get("/recomendacion/{titulo}")
def recomendacion(titulo):

    title=titulo.lower()

    #Validar que el titulo ingresado se encuentra en el dataframe
    if title not in moviesML['title'].values:
        return 'Película no encontrada! Prueba con: Star Wars, Back to the Future, Dracula, Cars, Batman, Superman, Toy Story'
    else:
        # Si el título esta en el df, encuentra su indice
        index = indicesML[title]

        # Obtiene las puntuaciones de similitud de las 5 peliculas más cercanas
        distances, indices_knn = nn.kneighbors(sparse_matrix[index], n_neighbors=6)  # indica que queremos encontrar las 6 peliculas más similares, incluyendo la pelicula dada

        # Obtiene los indices de las peliculas
        movie_indices = indices_knn[0][1:]  # Se omite el primer indice (la pelicula misma) con [1:]

        # Devuelve las 5 peliculas mas similares
        return moviesML['title'].iloc[movie_indices].tolist()