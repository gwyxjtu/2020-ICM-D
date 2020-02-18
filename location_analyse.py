import csv
from event_class import *
import networkx as nx
import matplotlib.pyplot as plt
import scipy as sp
import numpy as np

def dict_avg(my_dict):
	#求字典值的平均值
	l = len(my_dict)
	my_sum = sum(my_dict.values())
	return(my_sum/l)
def dict_max(my_dict):
	my_max = max(my_dict.values())
	return(my_max)
def new_clustering_analys(DF_adj, re_type):
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
	if(re_type == 1):
		return nx.clustering(G)#取平均，队伍或者队员都可以
	elif(re_type == 2):
		L = nx.normalized_laplacian_matrix(G)
		e = np.linalg.eigvals(L.A)
		#print("Largest eigenvalue:", max(e))#衡量什么同行网络
		return max(e)
	elif(re_type == 3):
		return nx.algebraic_connectivity(G)
	elif(re_type == 4):
		return(nx.reciprocity(G_i))
	elif(re_type == 5):
		return(nx.transitivity(G_i))
	elif(re_type == 6):
		return(nx.in_degree_centrality(G_i))
	elif(re_type == 7):
		return(nx.out_degree_centrality(G_i))
	elif(re_type == 8):
		try:
			return(nx.pagerank(G_i, alpha=0.9))
		except:
			return(0.01)
	elif(re_type == 9):
		try:
			return(nx.eigenvector_centrality(G))
		except:
			return(0.25)
	elif(re_type == 10):
		return(nx.average_neighbor_degree(G_i))

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
	if(re_type == 1):
		return dict_avg(nx.clustering(G))#取平均，队伍或者队员都可以
	elif(re_type == 2):
		L = nx.normalized_laplacian_matrix(G)
		e = np.linalg.eigvals(L.A)
		#print("Largest eigenvalue:", max(e))#衡量什么同行网络
		return max(e)
	elif(re_type == 3):
		return nx.algebraic_connectivity(G)
	elif(re_type == 4):
		return(nx.reciprocity(G_i))
	elif(re_type == 5):
		return(nx.transitivity(G_i))
	elif(re_type == 6):
		return(dict_max(nx.in_degree_centrality(G_i)))
	elif(re_type == 7):
		return(dict_max(nx.out_degree_centrality(G_i)))
	elif(re_type == 8):
		try:
			return(dict_avg(nx.pagerank(G, alpha=0.9)))
		except:
			return(0.01)
	elif(re_type == 9):
		try:
			return(dict_avg(nx.eigenvector_centrality(G)))
		except:
			return(0.25)
	elif(re_type == 10):
		return(dict_avg(nx.average_neighbor_degree(G_i)))
	print("-----------------")
	print(nx.closeness_centrality(G))#衡量星际球员
	print("-----------------")
	print(nx.pagerank(G, alpha=0.9))#衡量球员
	print("-----------------")
	print(nx.eigenvector_centrality(G))#衡量球员
	print("-----------------")
	print()#宏观的连通性
	print("-----------------")



def time_update(n,csv1,i):
	#遍历i*10的球队
	# n是之前的class，csv是具体的数据，i是遍历的次数
	n.update(csv1['TeamID'][50+i*10:60+i*10],csv1['OriginPlayerID'][50+i*10:60+i*10],csv1['DestinationPlayerID'][50+i*10:60+i*10],csv1['EventTime'][50+i*10:60+i*10],csv1['EventOrigin_x'][50+i*10:60+i*10],csv1['EventOrigin_y'][50+i*10:60+i*10],csv1['EventDestination_x'][50+i*10:60+i*10],csv1['EventDestination_y'][50+i*10:60+i*10])
	return n


def new_heat_plot(csv1):
	#微观球员图像函数
	#获得得分的方式
	shot_1 = [8, 7, 7, 9, 6, 15, 24, 11, 7, 13, 8, 7, 5, 6, 6]#, 1, 7, 10, 4, 5, 7]#, 10, 5, 8, 11, 11, 8, 4, 10, 12, 15, 5, 6, 10, 7]
	shot_2 = [10, 18, 18, 15, 12, 8, 4, 11, 26, 7, 10, 14, 15, 5, 4]#, 24, 13, 7, 24, 13, 17]#, 21, 20, 15, 6, 14, 13, 12, 21, 10, 10, 18, 10, 15, 8]
	csv2=pd.read_csv('matches.csv')
	score_1 = csv2['OwnScore'].tolist()[:21]
	score_2 = csv2['OpponentScore'].tolist()[:21]
	# for i in range(21):
	# 	tmp_1 = score_1.pop(0)
	# 	tmp_2 = score_2.pop(0)
	# 	score_1.append((tmp_1+1)/(tmp_2+1))
	# 	score_2.append((tmp_2+1)/(tmp_1+1))
	#score 用比例
	#只分析了前20场
	df_new = [csv1[csv1.MatchID == i] for i in range(1,39)]
	#按照比赛划分为多个小的dataframe文件
	#每一个队伍都应该有自己的分析,刚才分好的csv可以保证每次传进去的不会出现第三支队伍
	#每一场比赛分析一个oppo队伍和哈士奇队。
	cor_list = []
	cor_list_1 = []
	for i in range(10):
		tmp_1 = []
		tmp_2 = []
		for j in range(14):
			tmp_1.append([])
			tmp_2.append([])
		cor_list.append(tmp_1)
		cor_list_1.append(tmp_2)
	arg_list = ['clustering','in_degree_centrality','out_degree_centrality','pagerank','average_neighbor_degree']
	for o in [1,6,7,8,10]:

		final = []
		fin_1 = []
		fin_2 = []
		for i in range(1,16):
			fin_1.append([])
			fin_2.append([])
		df_index_1 = ['1']
		df_index_2 = ['1']
		#储存矩阵初始化


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

				#_,d_1,_,d_2 = n.get_avg_distance()

				d_1 = new_clustering_analys(adj_1,o)
				d_2 = new_clustering_analys(adj_2,o)

				# for k_1 in range(len(d_1)):
				# 	dog_player_score[k_1] += np.mean(d_1)
				# 	dog_player_times[k_1] += 1
				# for k_2 in range(len(d_2)):
				# 	oppo_player_score[k_2] += np.mean(d_2)
				# 	oppo_player_times[k_2] += 1


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

			break
		this_arg = arg_list.pop(0)
		#print(len(dog_player_score[i]),len(player_dog_list))
		fig=plt.figure()
		plt.subplots_adjust(wspace =0, hspace =1)#调整子图间距
		plt.subplot(2,1,1)
		plt_1 = pd.Series([dog_player_score[i]/(dog_player_times[i]+0.1) for i in range(len(player_dog_list))], index = [player_dog_list[i][-2:] for i in range(len(player_dog_list))], name = "Huskies")
		#if(match == 1):
		#	final.append(plt_1)
		plt_1.plot(kind='bar',title = "Huskies players' "+this_arg, ylim = (0,1.2*max(plt_1)))
		plt.subplot(2,1,2)
		plt_2 = pd.Series([oppo_player_score[i]/(oppo_player_times[i]+0.1) for i in range(len(player_oppo_list))], index = [player_oppo_list[i][-2:] for i in range(len(player_oppo_list))], name = this_team_name)
		plt_2.plot(kind='bar', title = this_team_name+' players\' '+this_arg, ylim = (0,1.2*max(plt_2)))
		final.append(plt_2)
		plt.show()
		#print(final)
		#plt.savefig("micro_plot/"+str(o)+".png")


def time_arg_plot(csv1):
	df_new = [csv1[csv1.MatchID == i] for i in range(1,39)]
	csv2=pd.read_csv('fullevents.csv')
	df_all = [csv2[csv2.MatchID == i] for i in range(1,39)]

	#按照比赛划分为多个小的dataframe文件
	#print(df_new[4])
	#队伍的列表
	#每一个队伍都应该有自己的分析,刚才分好的csv可以保证每次传进去的不会出现第三支队伍
	#每一场比赛分析一个oppo队伍和哈士奇队。

	#储存矩阵初始化
	name_list = ['normalized_laplacian_matrix','algebraic_connectivity','reciprocity','transitivity']
	for o in [2,3,4,5]:
		final = []
		fin_1 = []
		fin_2 = []
		for i in range(1,16):
			fin_1.append([])
			fin_2.append([])
		df_index_1 = []
		df_index_2 = []
		for match in range(1,15):#len(df_new)):
			#先要找到画竖线的地方
			shot_time1 = []
			shot_time2 = []
			for x in range(len(df_all[match-1])):
				if df_all[match-1]['EventType'].tolist()[x] == 'Shot':
					if(df_all[match-1]['MatchPeriod'].tolist()[x] == '2H'):
						if(df_all[match-1]['TeamID'].tolist()[x] == 'Huskies'):
							shot_time1.append(int(df_all[match-1]['EventTime'][x]/60)+45)
						else:
							shot_time2.append(int(df_all[match-1]['EventTime'][x]/60)+45)
					else:
						if(df_all[match-1]['TeamID'].tolist()[x] == 'Huskies'):
							shot_time1.append(int(df_all[match-1]['EventTime'][x]/60))
						else:
							shot_time2.append(int(df_all[match-1]['EventTime'][x]/60))
					



			team_name = list(set(csv1.TeamID))
			team_name.sort()

			this_team = list(set(df_new[match].TeamID))
			this_team.sort()
			#print(this_team)
			this_team_name = this_team[1]
			team_index = team_name.index(this_team_name)
			df_index_1.append("Huskies Game "+str(match))
			df_index_2.append(this_team_name+" Game "+str(match))
			#找到当前比赛的队伍在队伍列表中的位子
			#print("当前参数： ")
			print("比赛场次： ",end = "")
			print(match)
			n = model_50_passing(df_new[match]['TeamID'][:50].tolist(),df_new[match]['OriginPlayerID'][:50].tolist(),df_new[match]['DestinationPlayerID'][:50].tolist(),df_new[match]['EventTime'][:50].tolist(),df_new[match]['EventOrigin_x'][:50].tolist(),df_new[match]['EventOrigin_y'][:50].tolist(),df_new[match]['EventDestination_x'][:50].tolist(),df_new[match]['EventDestination_y'][:50].tolist())
			#初始化class
			
			time_list = []
			for ii in range(1, int((len(df_new[match]) - 50)/10)):
				#print(df_new[match]['MatchPeriod'].tolist()[ii*10])
				if(df_new[match]['MatchPeriod'].tolist()[ii*10+10] == '2H'):
					time_list.append(int(n.get_time_now()/60)+45)
				else:
					time_list.append(int(n.get_time_now()/60))
				n = time_update(n, df_new[match], ii)
				#更新时间列表，作为绘图的横轴
				adj_1,adj_2 = n.get_adj_mat()

				d_1 = clustering_analys(adj_1,o)
				fin_1[match-1].append(d_1.real)
				d_2 = clustering_analys(adj_2,o)
				fin_2[match-1].append(d_2.real)
			
			#break
			#plt_1 = pd.Series([dog_player_score[i]/(dog_player_times[i]+0.1) for i in range(len(player_dog_list))], index = [player_dog_list[i][-2:] for i in range(len(player_dog_list))], name = "Huskies")

			#plt_1.plot(kind='bar',title = "average Clustering coefficient in Huskies", ylim = (0,0.6))
			#plt_2 = pd.Series([oppo_player_score[i]/(oppo_player_times[i]+0.1) for i in range(len(player_oppo_list))], index = [player_oppo_list[i][-2:] for i in range(len(player_oppo_list))], name = this_team_name)
			fig=plt.figure(figsize=(10,5))
			tmp_name = name_list.pop(0)
			plt.ylabel(tmp_name)
			plt.xlabel("Time/(minutes)")
			for t in range(len(shot_time1)-1):
				plt.axvline(shot_time1[t],ls = '--', color = '#00CED1')
			for t in range(len(shot_time2)-1):
				plt.axvline(shot_time2[t],ls = '--', color = '#DC143C')
			print(time_list,fin_2[match-1])
			plt.ylim(0.8*min(min(fin_1[match-1]),min(fin_2[match-1])),1.2*max(max(fin_1[match-1]),max(fin_2[match-1])))
			#pd.DataFrame(fin_1[match-1]).to_csv(tmp_name+'_fin_1.csv')
			#pd.DataFrame(fin_2[match-1]).to_csv(tmp_name+'_fin_2.csv')
			pd.DataFrame(shot_time1).to_csv('shot_time1.csv')
			pd.DataFrame(shot_time2).to_csv('shot_time2.csv')
			#pd.DataFrame(time_list).to_csv(tmp_name+'time_list.csv')
			plt.plot(time_list, fin_1[match-1],'bo-',color = '#00CED1', label = 'Huskies')#,ylim = (0,1.2*max(fin_1[match-1])))
			plt.plot(time_list, fin_2[match-1],'bo-', color = '#DC143C',label = 'Opponent')#,ylim = (0,1.2*max(fin_2[match-1])))
			#plt_2.plot(kind='bar', title = "average Clustering coefficient in "+this_team_name, ylim = (0,0.6))
			plt.legend()

			#plt.show()
			plt.savefig("time_plot/time "+tmp_name+".png")
			break
def fake_plot():
	name_list = ['algebraic_connectivity']
	fin_1 = pd.read_csv('algebraic_connectivity_fin_1.csv')['0'].tolist()
	#print(fin_1['0'])
	fin_2 = pd.read_csv('algebraic_connectivity_fin_2.csv')['0'].tolist()
	shot_time1 = pd.read_csv('shot_time1.csv')['0'].tolist()
	shot_time2 = pd.read_csv('shot_time2.csv')['0'].tolist()
	time_list = pd.read_csv('reciprocitytime_list.csv')['0'].tolist()
	fig=plt.figure(figsize=(10,5))
	tmp_name = name_list.pop(0)
	plt.ylabel(tmp_name)
	plt.xlabel("Time/(minutes)")
	print(shot_time1[3])
	for t in range(len(shot_time1)-1):
		plt.axvline(shot_time1[t],ls = '--', color = '#00CED1')
	for t in range(len(shot_time2)-1):
		plt.axvline(shot_time2[t],ls = '--', color = '#DC143C')
	#print(time_list,fin_2)
	plt.ylim(0.8*min(min(fin_1),min(fin_2)),1.2*max(max(fin_1),max(fin_2)))

	plt.plot(time_list, fin_1,'bo-',color = '#00CED1', label = 'Huskies')#,ylim = (0,1.2*max(fin_1[match-1])))
	plt.plot(time_list, fin_2,'bo-', color = '#DC143C',label = 'Opponent')#,ylim = (0,1.2*max(fin_2[match-1])))
	#plt_2.plot(kind='bar', title = "average Clustering coefficient in "+this_team_name, ylim = (0,0.6))
	plt.legend()
	plt.show()
if __name__ == '__main__':
	csv1=pd.read_csv('passingevents.csv')
	# csv_full = pd.read_csv('fullevents.csv')
	# df_new = [csv_full[csv_full.MatchID == i] for i in range(1,39)]
	# df_new = df_new[]

	#clustering_analys(DF_adj,3)

	#get_shot_oppotinuty()
	#[8, 7, 7, 9, 6, 15, 24, 11, 7, 13, 8, 7, 5, 6, 6, 1, 7, 10, 4, 5, 7, 10, 5, 8, 11, 11, 8, 4, 10, 12, 15, 5, 6, 10, 7] [10, 18, 18, 15, 12, 8, 4, 11, 26, 7, 10, 14, 15, 5, 4, 24, 13, 7, 24, 13, 17, 21, 20, 15, 6, 14, 13, 12, 21, 10, 10, 18, 10, 15, 8]
	new_heat_plot(csv1)
	#time_arg_plot(csv1)
