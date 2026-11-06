# note: console is https://us-east-2.console.aws.amazon.com/ec2/home?region=us-east-2#Instances:v=3;$case=tags:true%5C,client:false;$regex=tags:false%5C,client:false;sort=desc:instanceState

# run python inside the uv project env (../pyproject.toml)
PY = uv run python

default:
	echo make targets: 'launch stop terminate wc_[ctr|nano|micro|hz|spork]'

# cluster instances are tagged Name=$(CLUSTER), so the targets below
# only touch them, not any other instances in the region

CLUSTER = hazsoup
K ?= 4

# list ids of cluster instances; append more --filters args to narrow it
LIST_IDS = aws ec2 describe-instances --output text \
	--query 'Reservations[].Instances[].InstanceId' \
	--filters Name=tag:Name,Values=$(CLUSTER)

# stop the running cluster instances (their disks are still billed)

stop:
	ids=`$(LIST_IDS) Name=instance-state-name,Values=pending,running`; \
	if [ -n "$$ids" ]; then aws ec2 stop-instances --instance-ids $$ids; \
	else echo no running $(CLUSTER) instances; fi

# delete the cluster instances and their disks for good

terminate:
	ids=`$(LIST_IDS) Name=instance-state-name,Values=pending,running,stopping,stopped`; \
	if [ -n "$$ids" ]; then aws ec2 terminate-instances --instance-ids $$ids; \
	else echo no $(CLUSTER) instances; fi

# launch a cluster of K instances, e.g., make launch K=8

launch:
	aws ec2 run-instances \
	    --image-id ami-0c803b171269e2d72 \
	    --instance-type t2.micro \
	    --security-groups launch-wizard-1 \
	    --key-name hazsoup \
	    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=$(CLUSTER)}]' \
	    --count $(K) > launch-log.json

launch_k: launch

# workers.json maps internal machine names to their external names.
# It's rebuilt every time, since instances get new external names
# whenever they are started.

.PHONY: workers.json
workers.json:
	aws ec2 describe-instances --filters Name=tag:Name,Values=$(CLUSTER) \
	    Name=instance-state-name,Values=running \
	    --query 'Reservations[].Instances[].[PrivateDnsName,PublicDnsName]' \
	> workers.json

# cluster test

wc_hz: workers.json
	$(PY) hazsoup.py cloud setup --local_files wc_hz.py
	$(PY) hazsoup.py fs put ../data/brown_nolines.txt brown.txt
	$(PY) hazsoup.py run map_reduce wc_hz.py WordCount brown.txt browncounts.txt
	$(PY) hazsoup.py fs get_merge browncounts.txt /tmp/brown_counts.txt
	head /tmp/brown_counts.txt
	egrep -w '(the|of|and|to|in|that|is|was|he)' /tmp/brown_counts.txt

wc_enron: workers.json
	$(PY) hazsoup.py cloud setup --local_files wc_hz.py
	$(PY) hazsoup.py fs put ~/Documents/data/enron/enron-clean.txt enron.txt
	time $(PY) hazsoup.py run map_reduce wc_hz.py WordCount enron.txt enroncounts.txt
	$(PY) hazsoup.py fs get_merge enroncounts.txt /tmp/enron_counts.txt
	head /tmp/enron_counts.txt
	egrep -w '(the|of|and|to|in|that|is|was|he)' /tmp/enron_counts.txt

# ~1Gb of text: the first 4 shards of the deduped-text-v1 corpus, one
# document per line

DEDUP_1G = ../data/deduped-1g.txt

$(DEDUP_1G):
	$(PY) parquet_to_lines.py $@ $(wordlist 1,4,$(sort $(wildcard deduped-text-v1/shard_*.parquet)))

wc_dedup_1g: workers.json $(DEDUP_1G)
	$(PY) hazsoup.py cloud setup --local_files wc_hz.py
	$(PY) hazsoup.py fs put $(DEDUP_1G) dedup1g.txt
	time $(PY) hazsoup.py run map_reduce wc_hz.py WordCount dedup1g.txt dedup1gcounts.txt
	$(PY) hazsoup.py fs get_merge dedup1gcounts.txt /tmp/dedup1g_counts.txt
	head /tmp/dedup1g_counts.txt
	egrep -w '(the|of|and|to|in|that|is|was|he)' /tmp/dedup1g_counts.txt

# local tests

wc_ctr:
	$(PY) wc_ctr.py

wc_nano:
	$(PY) -m fire wc_nano.py \
	  WordCount map_reduce --src ../data/brown_nolines.txt --dst /tmp/brown_counts.txt 
	head /tmp/brown_counts.txt
	egrep -w '(the|of|and|to|in|that|is|was|he)' /tmp/brown_counts.txt

wc_micro:
	$(PY) -m fire wc_micro.py \
	  WordCount map_reduce --src ../data/brown_nolines.txt --dst /tmp/brown_counts.txt 
	head /tmp/brown_counts.txt
	egrep -w '(the|of|and|to|in|that|is|was|he)' /tmp/brown_counts.txt

wc_spork:
	$(PY) wc_spork.py

