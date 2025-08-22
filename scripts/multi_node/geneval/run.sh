script=$1
export http_proxy=http://oversea-squid2.ko.txyun:11080 https_proxy=http://oversea-squid2.ko.txyun:11080 no_proxy=localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com
export HF_HOME=/m2v_intern/liuhenglin/code/video_gen/models
cd /m2v_intern/liuhenglin/code/video_gen/flow_grpo
bash /m2v_intern/liuhenglin/code/video_gen/flow_grpo/scripts/multi_node/geneval/${script}