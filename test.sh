SHARE=$(pwd)
IMAGE=$1
NAME=$(sudo docker run -d -v ${SHARE}:/host/Users -it ${IMAGE} /bin/bash)
echo '****************'
docker exec -it $NAME /bin/bash
# sudo docker exec -i $NAME ./ask sets/set1/a1.txt 3 2>/dev/null
# echo '****************'
# sudo docker exec -i $NAME ./answer sets/set1/a1.txt test_questions.txt
# echo '****************'
sudo docker stop $NAME >/dev/null
sudo docker rm $NAME