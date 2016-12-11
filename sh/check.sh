for i in ./train/m64h128d0*; 
do 
echo $i; 
grep 'targets' $i/log.txt | tail -n 1; 
grep "dev:" $i/log.txt; 
echo -n "lowest: "
grep "dev:" $i/log.txt | sort -nrk 3 | tail -n 1;
echo ""; 
done

for i in ./train/m64h128d0*;
do
p=$(grep "dev:" $i/log.txt | sort -nrk 3 | tail -n 1);
if [[ $p != "" ]]; then
    echo -n $i '';
    echo $p;
fi
done | sort -nk 4
