for i in ../train/m64*; 
do 
echo -n $i ""; 
grep 'SCORE_FORMAT' $i/log.recommend.txt | tail -n 1; 
done
