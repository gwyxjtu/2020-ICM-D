import csv
from event_class import *
import networkx as nx
import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
from pylab import *
import matplotlib.image as img

def dict_avg(my_dict):
	#求字典值的平均值
	l = len(my_dict)
	my_sum = sum(my_dict.values())
	return(my_sum/l)
def dict_max(my_dict):
	my_max = max(my_dict.values())
	return(my_max)
def nx_plt(DF_adj, locat_x, locat_y):
	#画图的函数
	labels = list(DF_adj.index)
	#print(DF_adj_1,DF_adj)
	#Network graph
	G = nx.DiGraph()

	G.add_nodes_from(labels)

	#Connect nodes
	for i in range(DF_adj.shape[0]):
	    col_label = DF_adj.columns[i]
	    for j in range(DF_adj.shape[1]):
	        row_label = DF_adj.index[j]
	        node = DF_adj.iloc[i,j]
	        if node != 0:
	            #print(node,DF_adj[labels[i]][labels[j]])
	            #print(node)
	            G.add_edge(col_label,row_label,weight = node*10)
	#Draw graph
	pos = {}
	for i,l in enumerate(labels):
		pos[l] = np.array([locat_y[i],locat_x[i]])

	nx.draw(G,pos,with_labels = True)
	print("---------")
	print(nx.spring_layout(G))
	#DRAWN GRAPH MATCHES THE GRAPH FROM WIKI
	plt.show()
	#Recreate adjacency matrix
	DF_re = pd.DataFrame(np.zeros([len(G.nodes()),len(G.nodes())]),index=G.nodes(),columns=G.nodes())
	for col_label,row_label in G.edges():
	    DF_re.loc[col_label,row_label] = 1
	    DF_re.loc[row_label,col_label] = 1
	# print(G.edges())
def nx_vector_plt(DF_adj, n):
	#画向量图使用
	kick_ball_location_x, kick_ball_location_y, get_ball_location_x, get_ball_location_y = n.get_dynamic_loction()
	labels = list(DF_adj.index)
	print(labels)
	c = randn(len(get_ball_location_x)) # arrow颜色
	for i in range(len(kick_ball_location_x)):
		kick_ball_location_x[i] = kick_ball_location_x[i] - get_ball_location_x[i]
		kick_ball_location_y[i] = kick_ball_location_y[i] - get_ball_location_y[i]
	#figure()
	#bgimg = img.imread('./111.png')
	#figimage(bgimg)
	xlim(0,100)
	ylim(0,100)
	quiver(get_ball_location_x, get_ball_location_y,kick_ball_location_x, kick_ball_location_y,c, scale_units='xy', scale=1) # 注意参数的赋值
	for i in range(len(get_ball_location_y)):
		annotate(labels[i][-2:],(get_ball_location_x[i],get_ball_location_y[i]))
	show()
	#savefig('test.png')
	#print(DF_adj_1,DF_adj)
	#Network graph

def avg_move_plot(n, csv1):
	#最后的表
	df_new = [csv1[csv1.MatchID == i] for i in range(1,39)]
	#labels = list(DF_adj.index)
	final = []
	for i in range(10):
		final.append([])
	for match in range(1,11):
		print("比赛场次： ",end = "")
		print(match)
		tmp = []
		for ii in range(int((len(df_new[match]) - 50)/10)):
			tmp.append([])
		for ii in range(1, int((len(df_new[match]) - 50)/10)):
			n = time_update(n, df_new[match], ii)
			adj_1,_ = n.get_adj_mat()
			kick_ball_location_x, kick_ball_location_y, get_ball_location_x, get_ball_location_y = n.get_dynamic_loction()
			#print([abs(kick_ball_location_x[i] - get_ball_location_x[i])+abs(kick_ball_location_y[i] - get_ball_location_y[i]) for i in range(len(get_ball_location_x))])
			tmp[ii].append([abs(kick_ball_location_x[i] - get_ball_location_x[i])+abs(kick_ball_location_y[i] - get_ball_location_y[i]) for i in range(len(get_ball_location_x))])
		
		for i in range(5):
			
			#print(final[match-1])
			#print(tmp)
			tmp_1 = 0
			for x in range(1, int((len(df_new[match]) - 50)/10) - 3):
				#print(x,i)
				tmp_1 += tmp[x][0][i]

			final[match-1].append(tmp_1/(x-1)) 
	pd.DataFrame(final ).to_csv("final.csv")
	print(final)

def clustering_analys(DF_adj, re_type):
	#测试参数的函数。re_type是返回值的类型
	labels = list(DF_adj.index)
	#print(DF_adj_1,DF_adj)
	#Network graph
	G = nx.Graph()
	G_i = nx.DiGraph()
	G.add_nodes_from(labels)
	G_i.add_nodes_from(labels)
	#Connect nodes
	for i in range(DF_adj.shape[0]):
	    col_label = DF_adj.columns[i]
	    for j in range(DF_adj.shape[1]):
	        row_label = DF_adj.index[j]
	        node = DF_adj.iloc[i,j]
	        
	        if node != 0:
	            #print(node,DF_adj[labels[i]][labels[j]])
	            #print(node)
	            G.add_edge(col_label,row_label,weight = node)
	            G_i.add_edge(col_label,row_label,weight = node)
	        # else:
	        #     G.add_edge(col_label,row_label,weight = 100000)
	        #     G_i.add_edge(col_label,row_label,weight = 100000)
	if(re_type == 1):
		return nx.clustering(G)
	elif(re_type == 2):
		return nx.clustering(G)
	# print(nx.clustering(G))#取平均，队伍或者队员都可以
	# print("-----------------")
	# print(nx.in_degree_centrality(G_i),nx.out_degree_centrality(G_i))#用来评价星际球员
	# print("-----------------")
	# print(nx.closeness_centrality(G))#衡量星际球员
	# print("-----------------")
	# print(nx.pagerank(G, alpha=0.9))#衡量球员
	# print("-----------------")
	# print(nx.eigenvector_centrality(G))#衡量球员
	# print("-----------------")
	# print(nx.algebraic_connectivity(G))#宏观的连通性
	# print("-----------------")
	# L = nx.normalized_laplacian_matrix(G)
	# e = np.linalg.eigvals(L.A)
	# print("Largest eigenvalue:", max(e))#衡量什么同行网络
	# print("-----------------")
	if(re_type == 3):
		#print(nx.attr_matrix(G_i))
		return(nx.reciprocity(G_i))
	if(re_type == 5):
		return(nx.eigenvector_centrality(G_i))
	if(re_type == 6):
		return(dict_max(nx.in_degree_centrality(G_i)))

def time_update(n,csv1,i):
	#遍历i*10的球队
	# n是之前的class，csv是具体的数据，i是遍历的次数
	n.update(csv1['TeamID'][50+i*10:60+i*10],csv1['OriginPlayerID'][50+i*10:60+i*10],csv1['DestinationPlayerID'][50+i*10:60+i*10],csv1['EventTime'][50+i*10:60+i*10],csv1['EventOrigin_x'][50+i*10:60+i*10],csv1['EventOrigin_y'][50+i*10:60+i*10],csv1['EventDestination_x'][50+i*10:60+i*10],csv1['EventDestination_y'][50+i*10:60+i*10])
	return n


def entire_plot(csv1):
	df_new = [csv1[csv1.MatchID == i] for i in range(1,39)]
	#按照比赛划分为多个小的dataframe文件
	#print(df_new[4])

	#队伍的列表


	#每一个队伍都应该有自己的分析,刚才分好的csv可以保证每次传进去的不会出现第三支队伍
	#每一场比赛分析一个oppo队伍和哈士奇队。
	final = []
	for match in range(1,22):#len(df_new)):
		team_name = list(set(csv1.TeamID))
		team_name.sort()

		this_team = list(set(df_new[match].TeamID))
		this_team.sort()
		#print(this_team)
		this_team_name = this_team[1]
		team_index = team_name.index(this_team_name)
		#找到当前比赛的队伍在队伍列表中的位子
		print("比赛场次： ",end = "")
		print(match)
		n = model_50_passing(df_new[match]['TeamID'][:50].tolist(),df_new[match]['OriginPlayerID'][:50].tolist(),df_new[match]['DestinationPlayerID'][:50].tolist(),df_new[match]['EventTime'][:50].tolist(),df_new[match]['EventOrigin_x'][:50].tolist(),df_new[match]['EventOrigin_y'][:50].tolist(),df_new[match]['EventDestination_x'][:50].tolist(),df_new[match]['EventDestination_y'][:50].tolist())
		player_oppo_list = set()
		player_dog_list = set()
		#本次比赛所有的球员列表
		
		for iii in range(len(df_new[match]['OriginPlayerID'])):
			#print(df_new[match]['TeamID'][iii],this_team_name)
			if(df_new[match]['TeamID'].tolist()[iii] == this_team_name):
				player_oppo_list.add(df_new[match]['OriginPlayerID'].tolist()[iii])
				player_oppo_list.add(df_new[match]['DestinationPlayerID'].tolist()[iii])
			else:
				player_dog_list.add(df_new[match]['OriginPlayerID'].tolist()[iii])
				player_dog_list.add(df_new[match]['DestinationPlayerID'].tolist()[iii])

		#球员评分列表
		player_dog_list = list(player_dog_list)
		player_dog_list.sort()
		player_oppo_list = list(player_oppo_list)
		player_oppo_list.sort()
		
		#额外维护一个行动次数的数组，用来给分数求平均
		dog_player_score = [0]*len(player_dog_list)
		dog_player_times = [0]*len(player_dog_list)
		oppo_player_score = [0]*len(player_oppo_list)
		oppo_player_times = [0]*len(player_oppo_list)


		#获取两个队伍的球员信息
		#print(player_dog_list,player_oppo_list)
		#clustering_analys()
		for ii in range(1, int((len(df_new[match]) - 50)/10)):
			n = time_update(n, df_new[match], ii)
			adj_1,adj_2 = n.get_adj_mat()
			d_1 = clustering_analys(adj_1,1)
			d_2 = clustering_analys(adj_2,1)

			for k_1 in d_1.keys():
				try:
					dog_player_score[player_dog_list.index(k_1)] += dict_avg(d_1)
					dog_player_times[player_dog_list.index(k_1)] += 1
				except  ValueError:
					#print(adj_1,adj_2)
					print(ii)
					print("not find1 "+k_1)
			for k_2 in d_2.keys():
				try:
					oppo_player_score[player_oppo_list.index(k_2)] += dict_avg(d_2)
					oppo_player_times[player_oppo_list.index(k_2)] += 1
				except  ValueError:
					print("not find2 "+k_2)
		#plt.subplot(2,1,1)
		plt_1 = pd.Series([dog_player_score[i]/(dog_player_times[i]+0.1) for i in range(len(player_dog_list))], index = [player_dog_list[i][-2:] for i in range(len(player_dog_list))], name = "Huskies")
		if(match == 1):
			final.append(plt_1)
		plt_1.plot(kind='bar',title = "average Clustering coefficient in Huskies", ylim = (0,0.6))
		plt.subplot(2,1,2)
		plt_2 = pd.Series([oppo_player_score[i]/(oppo_player_times[i]+0.1) for i in range(len(player_oppo_list))], index = [player_oppo_list[i][-2:] for i in range(len(player_oppo_list))], name = this_team_name)
		plt_2.plot(kind='bar', title = "average Clustering coefficient in "+this_team_name, ylim = (0,0.6))
		final.append(plt_2)
		#print(final)
		plt.show()
		#只画一场比赛
		break
		#需要写入csv,先生成dataframe。

	#pd.DataFrame(final).to_csv("Clustering_coefficient.csv")



if __name__ == '__main__':
	csv1=pd.read_csv('passingevents.csv')
	n = model_50_passing(csv1['TeamID'][:50],csv1['OriginPlayerID'][:50],csv1['DestinationPlayerID'][:50],csv1['EventTime'][:50],csv1['EventOrigin_x'][:50],csv1['EventOrigin_y'][:50],csv1['EventDestination_x'][:50],csv1['EventDestination_y'][:50])
	#n.update(csv1['TeamID'][50:160],csv1['OriginPlayerID'][50:160],csv1['DestinationPlayerID'][50:160],csv1['EventTime'][50:160],csv1['EventOrigin_x'][50:160],csv1['EventOrigin_y'][50:160],csv1['EventDestination_x'][50:160],csv1['EventDestination_y'][50:160])
	DF_adj,DF_adj_1 = n.get_adj_mat()
	# for i in range(30):
	# 	n = time_update(n,csv1,i)
	# 	DF_adj,DF_adj_1 = n.get_adj_mat()
	# 	nx_vector_plt(DF_adj, n)
	#nx_vector_plt(DF_adj, n)
	avg_move_plot(n,csv1)
	#print(n.get_avg_distance())
		#print(clustering_analys(DF_adj,5))

	
	player_location_1_x,player_location_1_y = n.get_location(1)

	#nx_plt(DF_adj,player_location_1_y,player_location_1_x)

	#print(player_location_1_x,player_location_1_y)
	


	#entire_plot(csv1)
	

	


