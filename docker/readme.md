Docker Desktop instalado y ejecutándose.

Ejecuta el siguiente comando en la raíz del proyecto:

`docker build -t prediccion-lluvia .`


Esto creará una imagen llamada `prediccion-lluvia`.

Reemplaza `/ruta/a/tu/carpeta/files` con la ruta absoluta donde tienes tus datos.

`docker run --rm -v "/ruta/a/tu/carpeta/files:/files" prediccion-lluvia`