cd image-processing-service
source env/bin/activate
hypercorn app:app --bind localhost:8000 -w 1 --reload
