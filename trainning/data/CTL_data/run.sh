#!/bin/bash
FILES=/Users/acewood/code_base/research/simpleserver/public
# for f in $FILES
# do
# 	echo '123' >> $f
# done
for i in 1 2 3 4 5 6
do
	group=`ls -d $FILES/[$i]* 2>/dev/null`
	count=0
	temp=/Users/acewood/code_base/research/simpleserver/public/total_$i
	touch -c $temp
	first=true
	for f in $group
	do
		# if [ "$first" = true ]
		# then
		# 	let "first=false"
		# 	head -n 2 $f > $temp
		# fi
		# sed -e '1,2d' < $f >> $temp
		# let "count++"
		# done
		cat $f >> $temp
		let "count++"
	done
	echo $count
done