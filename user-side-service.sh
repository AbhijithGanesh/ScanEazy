cd user-side-service
source env/bin/activate
hypercorn app:app --bind localhost:8001 -w 1 --reload
