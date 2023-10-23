##################################################################
##################################################################
# App para: Convertir una imagen en secuencias de cubos rubik
##################################################################
##################################################################


#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# [A] Importacion de librerias
#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from IPython.display import display
from sklearn.cluster import KMeans

from itertools import permutations

import warnings
warnings.filterwarnings('ignore')

import tempfile
import streamlit as st

#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# [B] Creacion de funciones internas utiles
#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&


#=================================================================
# Definir funcion de distancia euclidiana
#=================================================================

@st.cache_resource() # https://docs.streamlit.io/library/advanced-features/caching
def dist_e(l1,l2):
  return sum(
    [(l1[i]-l2[i])**2 for i in range(0,len(l1))]
    )**0.5
  
#=================================================================
# Crear funcion de convertir rgb a lab
#=================================================================

@st.cache_resource() # https://docs.streamlit.io/library/advanced-features/caching
def conv_rgb_lab(l_rgb):
  
  return cv2.cvtColor(
    np.array(
      [[[float(l_rgb[0])/255,float(l_rgb[1])/255,float(l_rgb[2])/255]]], 
      dtype=np.float32
      ), 
    cv2.COLOR_RGB2Lab
    )[0][0].tolist()


#=================================================================
# Crear funcion de convertir rgb a luv
#=================================================================

@st.cache_resource() # https://docs.streamlit.io/library/advanced-features/caching
def conv_rgb_luv(l_rgb):
  
  return cv2.cvtColor(
    np.array(
      [[[float(l_rgb[0])/255,float(l_rgb[1])/255,float(l_rgb[2])/255]]], 
      dtype=np.float32
      ), 
    cv2.COLOR_RGB2Luv
    )[0][0].tolist()


#=================================================================
# Crear funcion principal que retorna lista de imagenes (con color)
#=================================================================

@st.cache_resource() # https://docs.streamlit.io/library/advanced-features/caching
def img2rubik(
  Nombre_Imagen,
  N_cubos_ancho,
  Colores_usar = ['Naranjo','Verde','Blanco','Azul','Rojo','Amarillo'],
  Espacio_Colores = 'RGB' # puede ser RGB, LAB o LUV
):


  #-----------------------------------------------------------------
  # [00] Calulos preliminares
  #-----------------------------------------------------------------

  # capturar imagen segun parametro ingresado (nombre de archivo o matriz RGB)
  if(len(Nombre_Imagen)<60):
    imagen_rgb = cv2.cvtColor(cv2.imread(Nombre_Imagen), cv2.COLOR_BGR2RGB)
  else:
    imagen_rgb = Nombre_Imagen

  # determinar dimensiones
  alto,ancho,_ = imagen_rgb.shape

  # calcular cantidad de cubos de alto
  N_cubos_alto = int(N_cubos_ancho*(alto/ancho))

  # calcular cantidad de colores
  N_Colores = len(Colores_usar)
  
  #-----------------------------------------------------------------
  # [01] Aplicar resize de la imagen
  #-----------------------------------------------------------------

  imagen_rgb_esc = cv2.resize(imagen_rgb, (3*N_cubos_ancho,3*N_cubos_alto))

  #-----------------------------------------------------------------
  # [02] Aplicar reduccion de colores usando kmeans
  #-----------------------------------------------------------------

  # pasar lista de valores a tabla 
  data_imagen_rgb_esc = pd.DataFrame(
    np.reshape(
      imagen_rgb_esc, 
      (imagen_rgb_esc.shape[0]*imagen_rgb_esc.shape[1], 3)
      ),
    columns=['R','G','B']
    )

  # Aplicar kmeans segun cantidad de colores a utilizar
  cluster_KM6 = KMeans(n_clusters = N_Colores)
  cluster_KM6.fit(data_imagen_rgb_esc[['R','G','B']])

  # ahora se aplica a cada registro su respectivo centroide
  data_imagen_rgb_esc2 = data_imagen_rgb_esc.copy()
  for clusterId in range(N_Colores):
    
    centroide = cluster_KM6.cluster_centers_[clusterId]
    indices = cluster_KM6.labels_==clusterId
    
    data_imagen_rgb_esc2.loc[indices,'R2']= centroide[0].astype(np.uint8)
    data_imagen_rgb_esc2.loc[indices,'G2']= centroide[1].astype(np.uint8)
    data_imagen_rgb_esc2.loc[indices,'B2']= centroide[2].astype(np.uint8)


  # re-armar imagen
  imagen_rgb_esc_km=np.reshape(
    np.array(data_imagen_rgb_esc2[['R2','G2','B2']]), 
    (imagen_rgb_esc.shape[0], imagen_rgb_esc.shape[1], 3)
    ).astype(np.uint8)


  #-----------------------------------------------------------------
  # [03] Aplicar asignacion de colores mas cercanos de rubik
  #-----------------------------------------------------------------

  # crear df con listado de colores rubik
  df_color = pd.DataFrame({
    'color': ['Naranjo','Verde','Blanco','Azul','Rojo','Amarillo'],
    'RGB': [[255,165,0],[0,255,0],[255,255,255],[0,0,255],[255,0,0],[255,255,0]]
  })


  # crear nuevo df solo con colores que se usaran
  df_color2 = df_color[df_color['color'].isin(Colores_usar)]


  # Obtener todas las combinaciones de distinto orden manteniendo los 6 elementos
  combs = list(permutations(list(df_color2['RGB'])))

  # ver valores unicos en imagen resultante
  l_img = list(set(tuple(v) for m2d in imagen_rgb_esc_km for v in m2d))

  # crear df en blanco donde se iran acumulando las distancias
  df_asig = pd.DataFrame([])

  # recorrer cada combinacion
  for c in combs:
    
    # crear lista de distancias
    l_dist = []
    
    # recorrer cada terna RGB dentro de cada combinacion
    for i in range(len(c)):
      
      # calcular distancia dependiendo de espacio de colores a utilizar
      if(Espacio_Colores=='RGB'):
        
        dist = dist_e(
          c[i],
          l_img[i]
          )
        
      elif(Espacio_Colores=='LAB'):
        
        dist = dist_e(
          conv_rgb_lab(c[i]),
          conv_rgb_lab(l_img[i])
          )
        
      elif(Espacio_Colores=='LUV'):
        
        dist = dist_e(
          conv_rgb_luv(c[i]),
          conv_rgb_luv(l_img[i])
          )
      
      # agregar distancia
      l_dist.append(dist)
      
    # consolidar en df
    df_asig = pd.concat([
      df_asig,
      pd.DataFrame({
        'comb': [c],
        'img': [l_img],
        'dist_total': sum(l_dist)
      })
    ])
    

  # buscar el que tiene minima distancia
  mejor_asig = df_asig.loc[
    df_asig['dist_total']==min(df_asig['dist_total'])
    ].iloc[0]

  # crear df de mapping
  df_map = pd.DataFrame({
    'real': [list(x) for x in mejor_asig[1]],
    'rubik': mejor_asig[0]
  })


  # crear diccionario para mapear luego los colores
  dic_R = dict(
    map(
      lambda i,j : (i,j), 
      [x[0] for x in df_map['real']],
      [x[0] for x in df_map['rubik']]
      )
    )

  dic_G = dict(
    map(
      lambda i,j : (i,j), 
      [x[1] for x in df_map['real']],
      [x[1] for x in df_map['rubik']]
      )
    )

  dic_B = dict(
    map(
      lambda i,j : (i,j), 
      [x[2] for x in df_map['real']],
      [x[2] for x in df_map['rubik']]
      )
    )

  # df de imagen anterior escalada con RGB segun kmeans
  data_imagen_rgb_esc3 = data_imagen_rgb_esc2.copy()

  data_imagen_rgb_esc3['R3'] = data_imagen_rgb_esc3['R2'].apply(
    lambda x: dic_R[int(x)]
  )
  
  data_imagen_rgb_esc3['G3'] = data_imagen_rgb_esc3['G2'].apply(
    lambda x: dic_G[int(x)]
  )
  
  data_imagen_rgb_esc3['B3'] = data_imagen_rgb_esc3['B2'].apply(
    lambda x: dic_B[int(x)]
  )
  

  # re-armar imagen
  imagen_rgb_esc_km_rubik=np.reshape(
    np.array(data_imagen_rgb_esc3[['R3','G3','B3']]), 
    (imagen_rgb_esc.shape[0], imagen_rgb_esc.shape[1], 3)
    ).astype(np.uint8)

  # retornar entregables
  return [
    imagen_rgb,
    imagen_rgb_esc,
    imagen_rgb_esc_km,
    imagen_rgb_esc_km_rubik
    ]





#=================================================================
# Crear funcion para graficar una imagen (en colores)
#=================================================================

@st.cache_resource() # https://docs.streamlit.io/library/advanced-features/caching
def graf_1img(
  lista_imgs,
  n_imagen
):
  
  fig,ax = plt.subplots(figsize = (60,28))
  
  if(n_imagen==5):
    
    #ax.figure(figsize = (60,28))
    ax.imshow(lista_imgs[n_imagen-2],extent=[0,lista_imgs[n_imagen-2].shape[1],0,lista_imgs[n_imagen-2].shape[0]])
    ax.tick_params(axis='both', labelsize=0, length = 0)
    ax.set_xticks(np.arange(0,lista_imgs[n_imagen-2].shape[1],3))
    ax.set_yticks(np.arange(0,lista_imgs[n_imagen-2].shape[0],3))
    ax.grid(color = 'black', linestyle = '-', linewidth = 0.8)
    #ax.show()

  else:
    
    #ax.figure(figsize = (60,28))
    ax.imshow(lista_imgs[n_imagen-1])
    ax.axis('off') 
    #ax.show()
    
  # generar entregable
  return fig
    

#=================================================================
# Crear funcion para graficar el subplot de imagenes  (en colores)
#=================================================================

@st.cache_resource() # https://docs.streamlit.io/library/advanced-features/caching
def graf_subplot(
  lista_imgs,
  letra_titulo,
  ancho_subplot = 30 
):
  
  # determinar alto del subplot
  alto_subplot = int((lista_imgs[0].shape[0]/lista_imgs[0].shape[1])*ancho_subplot)


  # definir objeto 
  fig2 = plt.figure(constrained_layout=True,figsize=(ancho_subplot, alto_subplot))
  spec2 = gridspec.GridSpec(ncols=3, nrows=2, figure=fig2,wspace=0.05)
  ax1 = fig2.add_subplot(spec2[0, 0])
  ax2 = fig2.add_subplot(spec2[0, 1])
  ax3 = fig2.add_subplot(spec2[0, 2])
  ax4 = fig2.add_subplot(spec2[1, 0])
  ax5 = fig2.add_subplot(spec2[1, 1])

  # agregar imagenes 
  ax1.imshow(lista_imgs[0])
  ax1.axis('off') 

  ax2.imshow(lista_imgs[1])
  ax2.axis('off') 

  ax3.imshow(lista_imgs[2])
  ax3.axis('off') 

  ax4.imshow(lista_imgs[3])
  ax4.axis('off') 

  ax5.imshow(
    lista_imgs[3],
    extent=[0,lista_imgs[3].shape[1],0,lista_imgs[3].shape[0]]
    )
  ax5.tick_params(axis='both', labelsize=0, length = 0)
  ax5.set_xticks(np.arange(0,lista_imgs[3].shape[1],3))
  ax5.set_yticks(np.arange(0,lista_imgs[3].shape[0],3))
  ax5.grid(color = 'black', linestyle = '-', linewidth = 0.8)

  # definir titulos
  titulo1 = f'Imagen Original ({lista_imgs[0].shape[0]}x{lista_imgs[0].shape[1]})'
  titulo2 = f'Imagen Reducida ({lista_imgs[1].shape[0]}x{lista_imgs[1].shape[1]})'
  titulo3 = f'Reduccion Colores k-means'
  titulo4 = f'Match con colores rubik'
  titulo5 = f'Detalle cubos rubik ({int(lista_imgs[1].shape[0]/3)}x{int(lista_imgs[1].shape[1]/3)})'
  
  # insertar titulos 
  ax1.set_title(titulo1,size=letra_titulo)
  ax2.set_title(titulo2,size=letra_titulo)
  ax3.set_title(titulo3,size=letra_titulo)
  ax4.set_title(titulo4,size=letra_titulo)
  ax5.set_title(titulo5,size=letra_titulo)

  #plt.show()
  
  # generar entregable
  return fig2



#=================================================================
# Crear funcion principal que retorna lista de imagenes (en gris)
#=================================================================

@st.cache_resource() # https://docs.streamlit.io/library/advanced-features/caching
def img2rubik_gris(
  Nombre_Imagen,
  N_cubos_ancho,
  Colores_usar = ['Naranjo','Verde','Blanco','Azul','Rojo','Amarillo'],
  Espacio_Colores = 'LAB' # puede ser RGB, LAB o LUV
):


  #-----------------------------------------------------------------
  # [00] Calulos preliminares
  #-----------------------------------------------------------------
  
  # capturar imagen segun parametro ingresado (nombre de archivo o matriz RGB)
  if(len(Nombre_Imagen)<60):
    imagen_rgb = cv2.cvtColor(cv2.imread(Nombre_Imagen), cv2.COLOR_BGR2RGB)
  else:
    imagen_rgb = Nombre_Imagen
  
  # convertir imagen a gris
  imagen_gris = cv2.cvtColor(imagen_rgb, cv2.COLOR_RGB2GRAY)

  # determinar alto y ancho de imagen 
  alto,ancho = imagen_gris.shape

  # calcular cubos de alto
  N_cubos_alto = int(N_cubos_ancho*(alto/ancho))

  # calcular cantidad de colores
  N_Colores = len(Colores_usar)


  #-----------------------------------------------------------------
  # [01] Aplicar resize de la imagen
  #-----------------------------------------------------------------

  imagen_gris_esc = cv2.resize(imagen_gris, (3*N_cubos_ancho,3*N_cubos_alto))

  #-----------------------------------------------------------------
  # [02] Aplicar reduccion de colores usando kmeans
  #-----------------------------------------------------------------

  # pasar lista de valores a tabla 
  data_imagen_gris_esc = pd.DataFrame(
    np.reshape(
      imagen_gris_esc, 
      (imagen_gris_esc.shape[0]*imagen_gris_esc.shape[1])
      ),
    columns=['Gris']
    )
  
  # Aplicar kmeans segun cantidad de colores a utilizar
  cluster_KM6 = KMeans(n_clusters = N_Colores)
  cluster_KM6.fit(data_imagen_gris_esc[['Gris']])

  # ahora se aplica a cada registro su respectivo centroide
  data_imagen_gris_esc2 = data_imagen_gris_esc.copy()
  for clusterId in range(N_Colores):
    
    centroide = cluster_KM6.cluster_centers_[clusterId]
    indices = cluster_KM6.labels_==clusterId
    
    data_imagen_gris_esc2.loc[indices,'Gris2']= centroide[0].astype(np.uint8)

  # re-armar imagen
  imagen_gris_esc_km=np.reshape(
    np.array(data_imagen_gris_esc2[['Gris2']]), 
    (imagen_gris_esc.shape[0], imagen_gris_esc.shape[1])
    ).astype(np.uint8)


  #-----------------------------------------------------------------
  # [03] Aplicar asignacion de colores mas cercanos de rubik
  #-----------------------------------------------------------------

  # crear df con listado de colores rubik
  df_color = pd.DataFrame({
    'color': ['Naranjo','Verde','Blanco','Azul','Rojo','Amarillo'],
    'RGB': [[255,165,0],[0,255,0],[255,255,255],[0,0,255],[255,0,0],[255,255,0]]
  })
  
  # agregar columna distancia de cercania hacia el blanco
  if(Espacio_Colores=='RGB'):
    df_color['dist_b'] = df_color['RGB'].apply(
      lambda x: dist_e(x,[255,255,255])
    )
  elif(Espacio_Colores=='LAB'):
    df_color['dist_b'] = df_color['RGB'].apply(
      lambda x: dist_e(conv_rgb_lab(x),conv_rgb_lab([255,255,255]))
    )
  elif(Espacio_Colores=='LUV'):
    df_color['dist_b'] = df_color['RGB'].apply(
      lambda x: dist_e(conv_rgb_luv(x),conv_rgb_luv([255,255,255]))
    )

  # crear nuevo df solo con colores que se usaran y ordenar por distancia
  df_color2 = df_color[
    df_color['color'].isin(Colores_usar)
    ].sort_values('dist_b', ascending=False)
  
  # generar lista ordenada de valores de imagen 
  colores_img = list(np.sort(np.unique(np.reshape(
    imagen_gris_esc_km, 
    (imagen_gris_esc_km.shape[0]*imagen_gris_esc_km.shape[1])
    ))))
  
  # asignar a df nueva columna 
  df_color2['color_img'] = colores_img
  
  # crear diccionario para mapear luego los colores
  dic_Gris = dict(
    map(
      lambda i,j : (i,j), 
      df_color2['color_img'],
      df_color2['RGB']
      )
    )
  
  # df de imagen anterior escalada con RGB segun kmeans
  data_imagen_gris_esc3 = data_imagen_gris_esc2.copy()
  
  # aplicar diccionario
  data_imagen_gris_esc3['R3'] = data_imagen_gris_esc3['Gris2'].apply(
    lambda x: dic_Gris[int(x)][0]
  )
  
  data_imagen_gris_esc3['G3'] = data_imagen_gris_esc3['Gris2'].apply(
    lambda x: dic_Gris[int(x)][1]
  )
  
  data_imagen_gris_esc3['B3'] = data_imagen_gris_esc3['Gris2'].apply(
    lambda x: dic_Gris[int(x)][2]
  )
  
  # re-armar imagen
  imagen_gris_esc_km_rubik=np.reshape(
    np.array(data_imagen_gris_esc3[['R3','G3','B3']]), 
    (imagen_gris_esc.shape[0], imagen_gris_esc.shape[1], 3)
    ).astype(np.uint8)
  
  # retornar entregables
  return [
    imagen_rgb,
    imagen_gris,
    imagen_gris_esc,
    imagen_gris_esc_km,
    imagen_gris_esc_km_rubik
    ]





#=================================================================
# Crear funcion para graficar una imagen (en gris)
#=================================================================

@st.cache_resource() # https://docs.streamlit.io/library/advanced-features/caching
def graf_1img_gris(
  lista_imgs,
  n_imagen
):
  
  fig3,ax = plt.subplots(figsize = (60,28))

  if(n_imagen==6):
    
    #ax.figure(figsize = (60,28))
    ax.imshow(lista_imgs[n_imagen-2],extent=[0,lista_imgs[n_imagen-2].shape[1],0,lista_imgs[n_imagen-2].shape[0]])
    ax.tick_params(axis='both', labelsize=0, length = 0)
    ax.set_xticks(np.arange(0,lista_imgs[n_imagen-2].shape[1],3))
    ax.set_yticks(np.arange(0,lista_imgs[n_imagen-2].shape[0],3))
    ax.grid(color = 'black', linestyle = '-', linewidth = 0.8)
    #ax.show()

  elif(n_imagen in [2,3,4]):
    
    #ax.figure(figsize = (60,28))
    ax.imshow(lista_imgs[n_imagen-1],cmap='gray')
    ax.axis('off') 
    #ax.show()
  
  else:
    
    #ax.figure(figsize = (60,28))
    ax.imshow(lista_imgs[n_imagen-1])
    ax.axis('off') 
    #ax.show()
      
  # generar entregable
  return fig3
    
    

#=================================================================
# Crear funcion para graficar el subplot de imagenes  (en colores)
#=================================================================

@st.cache_resource() # https://docs.streamlit.io/library/advanced-features/caching
def graf_subplot_gris(
  lista_imgs,
  letra_titulo,
  ancho_subplot = 30 
):
  
  # determinar alto del subplot
  alto_subplot = int((lista_imgs[0].shape[0]/lista_imgs[0].shape[1])*ancho_subplot)


  # definir objeto 
  fig2 = plt.figure(constrained_layout=True,figsize=(ancho_subplot, alto_subplot))
  spec2 = gridspec.GridSpec(ncols=3, nrows=2, figure=fig2,wspace=0.05)
  ax1 = fig2.add_subplot(spec2[0, 0])
  ax2 = fig2.add_subplot(spec2[0, 1])
  ax3 = fig2.add_subplot(spec2[0, 2])
  ax4 = fig2.add_subplot(spec2[1, 0])
  ax5 = fig2.add_subplot(spec2[1, 1])
  ax6 = fig2.add_subplot(spec2[1, 2])

  # agregar imagenes 
  ax1.imshow(lista_imgs[0])
  ax1.axis('off') 

  ax2.imshow(lista_imgs[1],cmap='gray')
  ax2.axis('off') 

  ax3.imshow(lista_imgs[2],cmap='gray')
  ax3.axis('off') 

  ax4.imshow(lista_imgs[3],cmap='gray')
  ax4.axis('off') 
  
  ax5.imshow(lista_imgs[4])
  ax5.axis('off') 

  ax6.imshow(
    lista_imgs[4],
    extent=[0,lista_imgs[4].shape[1],0,lista_imgs[4].shape[0]]
    )
  ax6.tick_params(axis='both', labelsize=0, length = 0)
  ax6.set_xticks(np.arange(0,lista_imgs[4].shape[1],3))
  ax6.set_yticks(np.arange(0,lista_imgs[4].shape[0],3))
  ax6.grid(color = 'black', linestyle = '-', linewidth = 0.8)

  # definir titulos
  titulo1 = f'Imagen Original ({lista_imgs[0].shape[0]}x{lista_imgs[0].shape[1]})'
  titulo2 = f'Imagen en B&N'
  titulo3 = f'Imagen Reducida ({lista_imgs[1].shape[0]}x{lista_imgs[1].shape[1]})'
  titulo4 = f'Reduccion Colores k-means'
  titulo5 = f'Match con colores rubik'
  titulo6 = f'Detalle cubos rubik ({int(lista_imgs[1].shape[0]/3)}x{int(lista_imgs[1].shape[1]/3)})'
  
  # insertar titulos 
  ax1.set_title(titulo1,size=letra_titulo)
  ax2.set_title(titulo2,size=letra_titulo)
  ax3.set_title(titulo3,size=letra_titulo)
  ax4.set_title(titulo4,size=letra_titulo)
  ax5.set_title(titulo5,size=letra_titulo)
  ax6.set_title(titulo6,size=letra_titulo)

  #plt.show()
  
  # generar entregable
  return fig2



#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# [C] Generacion de la App
#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

# https://streamlit-emoji-shortcodes-streamlit-app-gwckff.streamlit.app/

st.set_page_config(layout='wide')

# autoria 
st.sidebar.markdown('**Autor :point_right: [Sebastian Barrera](https://www.linkedin.com/in/sebasti%C3%A1n-nicolas-barrera-varas-70699a28)**')

# subir archivo 
Archivo = st.sidebar.file_uploader('Subir Imagen jpg',type=['jpg'])

# colocar separador para mostrar en sidebar otras cosas
st.sidebar.markdown('---')

# titulo inicial 
st.markdown('## Tu Imagen en Cubos Rubik :large_red_square: :large_blue_square: :large_orange_square: :large_yellow_square: :large_green_square: 	:white_large_square:')




#_____________________________________________________________________________
# comenzar a desplegar app una vez ingresado el archivo

if Archivo:  
  
  # Deglegar en sidebar opciones de seleccion de parametros
  
  # Trabajar en gris o en colores
  st_rgb_gris = st.sidebar.radio(
    label='Trabajar imagen en Blanco y Negro?',
    options=['No','Si'],
    horizontal=True,
    )
    
  # Numero de cubos de ancho
  st_N_cubos_ancho = st.sidebar.slider(
    label='Cantidad de cubos a usar en ancho',
    min_value=10,
    max_value=50,
    value=25,
    step=1
  )
  
  # Colores a utilizar
  st_Colores_usar = st.sidebar.multiselect(
    label='Colores Rubik a usar:',
    options=['Naranjo','Verde','Blanco','Azul','Rojo','Amarillo'],
    default=['Naranjo','Verde','Blanco','Azul','Rojo','Amarillo']
    )
  
  # Espacio de Colores
  st_Espacio_Colores = st.sidebar.radio(
    label='Espacio de Colores a usar:',
    options=['RGB','LAB','LUV'],
    horizontal=True,
    )
  
  
  # rescatar ubicacion del archivo para cargarlo
  with tempfile.NamedTemporaryFile(delete=False) as archivo_temporal:
    archivo_temporal.write(Archivo.getvalue())
    st_Nombre_Imagen = archivo_temporal.name
    
    
  # Aplicar funcion dependiendo de parametro ingresado
  if(st_rgb_gris=='No'):
    
    mi_lista_imgs = img2rubik(
      Nombre_Imagen = st_Nombre_Imagen, 
      N_cubos_ancho = st_N_cubos_ancho,
      Colores_usar = st_Colores_usar,
      Espacio_Colores = st_Espacio_Colores
    )
    
  else: 
    
    mi_lista_imgs = img2rubik_gris(
      Nombre_Imagen = st_Nombre_Imagen, 
      N_cubos_ancho = st_N_cubos_ancho,
      Colores_usar = st_Colores_usar,
      Espacio_Colores = st_Espacio_Colores
    )
    
  
  # Titulo de subplot de imagenes
  st.markdown('### A. Secuencia de transformaciones de imagenes')
  
  
  # generar subplot de imagenes
  if(st_rgb_gris=='No'):
    
    st_fig_suplot = graf_subplot(
      lista_imgs = mi_lista_imgs,
      letra_titulo = 35,
      ancho_subplot = 30 
      )
    
  else: 
    
    st_fig_suplot = graf_subplot_gris(
      lista_imgs = mi_lista_imgs,
      letra_titulo = 35,
      ancho_subplot = 30 
      )
    
  # mostrar imagen 
  st.pyplot(st_fig_suplot)
  

  # Titulo de plot de una imagen en particular
  st.markdown('### ')
  st.markdown('### B. Ver una imagen en particular')
  
  
  
  # generar opciones de seleccion de imagenes
  if(st_rgb_gris=='No'):
    
    opciones_imgs = [
      '1. Original',
      '2. Reducida',
      '3. K-means',
      '4. Match Rubik',
      '5. Grilla cubos'
    ]
    
  else: 
    
    opciones_imgs = [
      '1. Original',
      '2. Blanco y Negro',
      '3. Reducida',
      '4. K-means',
      '5. Match Rubik',
      '6. Grilla cubos'
    ]
    
    
  # listar opciones
  st_Seleccion_img = st.radio(
    label='Selecciona Imagen:',
    options=opciones_imgs,
    horizontal=True,
    )
  
  
  # calcular imagen segun seleccion indicada 
  if(st_rgb_gris=='No'):
    
    st_fig_img = graf_1img(
      lista_imgs = mi_lista_imgs,
      n_imagen = opciones_imgs.index(st_Seleccion_img)+1
      )
    
  else: 
    
    st_fig_img = graf_1img_gris(
      lista_imgs = mi_lista_imgs,
      n_imagen = opciones_imgs.index(st_Seleccion_img)+1
      )
    
  # mostrar imagen 
  st.pyplot(st_fig_img)






#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

# !streamlit run App_Img2Rubik_v1.py

# para obtener TODOS los requerimientos de librerias que se usan
# !pip freeze > requirements.txt


# para obtener el archivo "requirements.txt" de los requerimientos puntuales de los .py
# !pipreqs "/Seba/Actividades Seba/Programacion Python/25_Imagenes a Rubik (09-10-23)/App/"

# Video tutorial para deployar una app en streamlitcloud
# https://www.youtube.com/watch?v=HKoOBiAaHGg&ab_channel=Streamlit



