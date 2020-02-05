# Example Provenance Filtering Analytic
This directory contains an example provenance analytic which uses the front end gRPC API as well as a simple REST API to make calls to the FAISS
index.  The example code can be used to demonstrate the general workflow of a provenance filtering analytic, though you will need to make changes to
have it fit your implementation.

Test Command
 ```
 # Terminal 1
 # Run command from this directory
 docker-compose up

 # Terminal 2
 # Run command from this directory
 python -m medifor -s $PWD -t /input  provenance filter $PWD/assets/Koala.jpg
 ```

Modify the mount path for `filtersvc.py` in the `docker-compose.yml` to point to your own images to run additional images through the example (It's a toy example though, so the index is jsut random).   

Please modify the second volume of the filter service in the `docker-compose.yml` file to point to the proper image dataset world folder.

Please Download `index_1.faiss`, `index_2.faiss`, and `index_3.faiss` index example files from: https://drive.google.com/drive/folders/1fm23zXALzrWu-KIFmFdw0PMxnlHxX8a3?usp=sharing