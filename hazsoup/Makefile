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
	aws ec2 describe-instances | \
	    grep -i publicdns | cut -d\" -f4 | uniq | grep com > workers.txt

# cluster test

wc_hz:
	python3 hazsoup.py cloud setup --local_files wc_hz.py	
	python3 hazsoup.py fs put ../data/bluecorpus.txt blue.txt
	python3 hazsoup.py run map_reduce wc_hz.py WordCount blue.txt bluecounts.txt
	python3 hazsoup.py fs get_merge bluecounts.txt bluecounts.txt
	grep -w this bluecounts.txt | head -10

# local tests

wc_nano_map_only:
	python3 -m fire wc_nano.py WordCount map_only ../data/bluecorpus.txt bluecounts.txt 
	grep -w this bluecounts.txt | head -10

wc_nano:
	python3 -m fire wc_nano.py WordCount map_reduce ../data/bluecorpus.txt bluecounts.txt 
	grep -w this bluecounts.txt | head -10

wc_micro_map_only:
	python3 -m fire wc_micro.py WordCount map_only ../data/bluecorpus.txt bluecounts.txt 
	grep -w this bluecounts.txt | head -10

wc_micro:
	python3 -m fire wc_micro.py WordCount map_reduce ../data/bluecorpus.txt bluecounts.txt 
	grep -w this bluecounts.txt | head -10

wc_spork:
	python3 wc_spork.py

