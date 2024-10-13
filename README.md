# DHM-UHT
This is a minimalist and high readability version of the DHM-UHT ( removing all details that affect readability such as verbose, checkpoint, etc.). These files include the necessary steps for unsupervised meta-learning and stability analysis of model training.

python omniglot_main.py --algo=MAML --clustering=DBSCAN
python miniimagenet_main.py --algo=MAML --clustering=DBSCAN
python domainnet_main.py --algo=ANIL --clustering=DBSCAN

Optional parameters are MAML/ANIL, DBSCAN/Kmeans

The full code and scripts will be released after the paper is accepted. Thank you for your understanding.
