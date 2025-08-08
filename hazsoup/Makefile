# note: console is https://us-east-2.console.aws.amazon.com/ec2/home?region=us-east-2#Instances:v=3;$case=tags:true%5C,client:false;$regex=tags:false%5C,client:false;sort=desc:instanceState

# workers.json maps internal machine names to their external names
workers.json:
	aws ec2 describe-instances \
	--query Reservations[1].Instances[].NetworkInterfaces[].PrivateIpAddresses[0].[PrivateDnsName,Association.PublicDnsName] \
	> workers.json

stop:
	aws ec2 stop-instances --instance-ids `aws ec2 describe-instances | grep -i instanceid | cut -d\" -f4`

launch:
	aws ec2 run-instances \
	    --image-id ami-0c803b171269e2d72 \
	    --instance-type t2.micro \
	    --security-groups launch-wizard-1 \
	    --key-name hazsoup \
	    --count 4 > launch-log.json

# cluster test

wc_hz:
	python3 hazsoup.py cloud setup --local_files wc_hz.py	
	python3 hazsoup.py fs put ../data/brown_nolines.txt brown.txt
	python3 hazsoup.py run map_reduce wc_hz.py WordCount brown.txt browncounts.txt
	python3 hazsoup.py fs get_merge browncounts.txt /tmp/brown_counts.txt
	head /tmp/brown_counts.txt
	egrep -w '(the|of|and|to|in|that|is|was|he)' /tmp/brown_counts.txt

# local tests

wc_ctr:
	python3 wc_ctr.py

wc_nano:
	python3 -m fire wc_nano.py \
	  WordCount map_reduce --src ../data/brown_nolines.txt --dst /tmp/brown_counts.txt 
	head /tmp/brown_counts.txt
	egrep -w '(the|of|and|to|in|that|is|was|he)' /tmp/brown_counts.txt

wc_micro:
	python3 -m fire wc_micro.py \
	  WordCount map_reduce --src ../data/brown_nolines.txt --dst /tmp/brown_counts.txt 
	head /tmp/brown_counts.txt
	egrep -w '(the|of|and|to|in|that|is|was|he)' /tmp/brown_counts.txt

wc_spork:
	python3 wc_spork.py

