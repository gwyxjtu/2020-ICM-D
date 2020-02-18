# 进出的功能
# 单个队伍
# 按照次数好一点，按照时间会丢失信息
# 出现传给对面球员
import numpy as np
import pandas as pd



class model_50_passing:
	def __init__(self, TeamID, OriginPlayerID, DestinationPlayerID, EventTime, EventOrigin_x, EventOrigin_y, EventDestination_x, EventDestination_y):
		# 初始化的都是长度为50的数组
		# self里面的变量有，两个队的临界矩阵（带权重），两个队每个队员的位置（取平均），当前模型时差，时序行为列表

		self.time_now = EventTime[10]
		team = list(set(TeamID))#球队名字
		if(len(team) == 1):
			team.append(team[0])
		team.sort()
		#print(team.sort())
		self.team = team
		#self.time_gap = EventTime[len(EventTime)-1] - EventTime[0]    
		#先循环生成球员id
		tmp_set_1 = set()
		tmp_set_2 = set()
		for t_id,o_id,d_id in zip(TeamID, OriginPlayerID, DestinationPlayerID):

			if t_id == team[0]:
				tmp_set_1.add(o_id)
				tmp_set_1.add(d_id)
			elif t_id == team[1]:
				tmp_set_2.add(o_id)
				tmp_set_2.add(d_id)
			player_id_1 = list(tmp_set_1)
			player_id_2 = list(tmp_set_2)
		self.adj_mat_1 = [[0] * len(player_id_1) for i in range(len(player_id_1))]
		self.adj_mat_2 = [[0] * len(player_id_2) for i in range(len(player_id_2))]
		#加一个平均接球位置和平均踢球位置
		self.get_ball_location_x = []
		self.get_ball_location_y = []
		self.kick_ball_location_x = []
		self.kick_ball_location_y = []
		#额外加上distance变量
		self.player_location_1_x = []
		self.player_location_1_y = []
		self.player_location_2_x = []
		self.player_location_2_y = []
		self.distance_1_x = []
		self.distance_1_y = []
		self.distance_2_x = []
		self.distance_2_y = []
		for i in range(len(player_id_1)):
			self.player_location_1_x.append([])
			self.player_location_1_y.append([])
			self.distance_1_x.append([])
			self.distance_1_y.append([])
			self.get_ball_location_x.append([])
			self.get_ball_location_y.append([])
			self.kick_ball_location_x.append([])
			self.kick_ball_location_y.append([])
		for i in range(len(player_id_2)):
			self.player_location_2_x.append([])
			self.player_location_2_y.append([])
			self.distance_2_x.append([])
			self.distance_2_y.append([])

		#为了维护时间这个变量，我们除过邻接矩阵，必额外用一个时序的行为队列结构来维护。
		self.event_list = [TeamID, OriginPlayerID, DestinationPlayerID, EventTime, EventOrigin_x, EventOrigin_y, EventDestination_x, EventDestination_y]
		#print(len(self.event_list))

		#临界矩阵权重

		for i in range(50):
			if TeamID[i] == team[0]:
				#边的权重++
				self.adj_mat_1[player_id_1.index(OriginPlayerID[i])][player_id_1.index(DestinationPlayerID[i])] += 1
				#储存球员位置信息
				#print(self.player_location_1_x)
				self.player_location_1_x[player_id_1.index(OriginPlayerID[i])].append(EventOrigin_x[i])
				#print(player_id_1.index(OriginPlayerID[i]))
				#print(self.player_location_1_x)
				self.player_location_1_y[player_id_1.index(OriginPlayerID[i])].append(EventOrigin_y[i])
				self.player_location_1_x[player_id_1.index(DestinationPlayerID[i])].append(EventDestination_x[i])
				self.player_location_1_y[player_id_1.index(DestinationPlayerID[i])].append(EventDestination_y[i])


				self.kick_ball_location_x[player_id_1.index(OriginPlayerID[i])].append(EventOrigin_x[i])
				self.kick_ball_location_y[player_id_1.index(OriginPlayerID[i])].append(EventOrigin_y[i])
				self.get_ball_location_x[player_id_1.index(DestinationPlayerID[i])].append(EventDestination_x[i])
				self.get_ball_location_y[player_id_1.index(DestinationPlayerID[i])].append(EventDestination_y[i])


				if(len(self.player_location_1_x[player_id_1.index(DestinationPlayerID[i])]) > 1):
					self.distance_1_x[player_id_1.index(DestinationPlayerID[i])].append(abs(EventDestination_x[i] - self.player_location_1_x[player_id_1.index(DestinationPlayerID[i])][-2]))
				if(len(self.player_location_1_y[player_id_1.index(DestinationPlayerID[i])]) > 1):
					self.distance_1_y[player_id_1.index(DestinationPlayerID[i])].append(abs(EventDestination_y[i] - self.player_location_1_y[player_id_1.index(DestinationPlayerID[i])][-2]))
			elif TeamID[i] == team[1]:
				#边的权重++
				self.adj_mat_2[player_id_2.index(OriginPlayerID[i])][player_id_2.index(DestinationPlayerID[i])] += 1
				#储存球员位置信息
				self.player_location_2_x[player_id_2.index(OriginPlayerID[i])].append(EventOrigin_x[i])
				self.player_location_2_y[player_id_2.index(OriginPlayerID[i])].append(EventOrigin_y[i])
				self.player_location_2_x[player_id_2.index(DestinationPlayerID[i])].append(EventDestination_x[i])
				self.player_location_2_y[player_id_2.index(DestinationPlayerID[i])].append(EventDestination_y[i])
				if(len(self.player_location_2_x[player_id_2.index(DestinationPlayerID[i])]) > 1):
					self.distance_2_x[player_id_2.index(DestinationPlayerID[i])].append(abs(EventDestination_x[i] - self.player_location_2_x[player_id_2.index(DestinationPlayerID[i])][-2]))
				if(len(self.player_location_2_y[player_id_2.index(DestinationPlayerID[i])]) > 1):
					self.distance_2_y[player_id_2.index(DestinationPlayerID[i])].append(abs(EventDestination_y[i] - self.player_location_2_y[player_id_2.index(DestinationPlayerID[i])][-2]))
		# print(self.adj_mat_1)
		# print(player_id_1)
		self.player_id_1 = player_id_1
		self.player_id_2 = player_id_2
		self.adj_mat_1_final = pd.DataFrame(self.adj_mat_1, columns = player_id_1, index = player_id_1)
		self.adj_mat_2_final = pd.DataFrame(self.adj_mat_2, columns = player_id_2, index = player_id_2)
		#print(self.player_location_1_x)
		tmp = 0
		for i in range(len(player_id_1)):
			tmp = self.player_location_1_x.pop(0)
			#print(self.player_location_1_x)
			self.player_location_1_x.append(np.mean(tmp))
			tmp = self.player_location_1_y.pop(0)
			self.player_location_1_y.append(np.mean(tmp))
			#新家的kick和get参数同样需要需要平均操作
			#--------------------------------
			tmp = self.kick_ball_location_x.pop(0)
			self.kick_ball_location_x.append(np.mean(tmp))
			tmp = self.kick_ball_location_y.pop(0)
			self.kick_ball_location_y.append(np.mean(tmp))
			tmp = self.get_ball_location_x.pop(0)
			self.get_ball_location_x.append(np.mean(tmp))
			tmp = self.get_ball_location_y.pop(0)
			self.get_ball_location_y.append(np.mean(tmp))
			#---------------------------------
			
		for i in range(len(player_id_2)):
			tmp = self.player_location_2_x.pop(0)
			self.player_location_2_x.append(np.mean(tmp))
			tmp = self.player_location_2_y.pop(0)
			self.player_location_2_y.append(np.mean(tmp))
		#print(self.adj_mat_1_final)
		#print(self.player_location_1_x)
	
	#def get_index():
	def get_dynamic_loction(self):
		#这里只维护了队伍1的数据，仅仅用来作图使用
		return self.kick_ball_location_x, self.kick_ball_location_y, self.get_ball_location_x, self.get_ball_location_y

	def get_adj_mat(self):
		#邻接矩阵，两只队伍的
		return self.adj_mat_1_final,self.adj_mat_2_final

	def get_timegap(self):
		return self.time_gap

	def get_location(self,team_num):
		#print(self.team[0][:4])
		if(team_num == 1):
			return self.player_location_1_x,self.player_location_1_y
		else:
			return self.player_location_2_x,self.player_location_2_y
	def get_avg_distance(self):
		player_location_1_x,player_location_1_y = self.get_sum_distance(1)
		player_location_2_x,player_location_2_y = self.get_sum_distance(2)
		DF_adj,DF_adj_1 = self.get_adj_mat()
		players_1 = list(DF_adj.index)
		players_2 = list(DF_adj_1.index)
		ans1_x = []
		ans1_y = []
		ans2_x = []
		ans2_y = []
		for i in range(len(players_1)):
			tmp1 = player_location_1_x.pop(0)
			tmp2 = player_location_1_y.pop(0)
			if(len(tmp1)>0):
				ans1_x.append(np.mean(tmp1))
				ans1_y.append(np.mean(tmp2))
			else:
				ans1_x.append(0)
				ans1_y.append(0)
		for i in range(len(players_2)):
			tmp1 = player_location_2_x.pop(0)
			tmp2 = player_location_2_y.pop(0)
			if(len(tmp1)>0):
				ans2_x.append(np.mean(tmp1))
				ans2_y.append(np.mean(tmp2))
			else:
				ans2_x.append(0)
				ans2_y.append(0)
		return ans1_x,ans1_y,ans2_x,ans2_y

	def get_sum_distance(self,team_num):
		if(team_num == 1):
			return self.distance_1_x,self.distance_1_y
		else:
			return self.distance_2_x,self.distance_2_y
	def get_time_now(self):
		return self.time_now
	def update(self, TeamID, OriginPlayerID, DestinationPlayerID, EventTime, EventOrigin_x, EventOrigin_y, EventDestination_x, EventDestination_y):
		#传进来的各个数据长度必须一致！
		tmp_list = [TeamID.tolist(), OriginPlayerID.tolist(), DestinationPlayerID.tolist(), EventTime.tolist(), EventOrigin_x.tolist(), EventOrigin_y.tolist(), EventDestination_x.tolist(), EventDestination_y.tolist()]
		#print(len(self.event_list+tmp_list))
		#print(tmp_list)
		#print(self.event_list[4])
		for j in range(len(self.event_list)):
			for i in range(50-len(TeamID)):
			#print(len(self.event_list[j]))
				self.event_list[j][i] = self.event_list[j][i+len(TeamID)]
			#print(len(self.event_list[j]))
			for i in range(len(TeamID)):
				#print(tmp_list[j])
				self.event_list[j][i+50-len(TeamID)] = tmp_list[j][i]
		#print("----------")
		#print(self.event_list[4])
		#print(self.event_list)
		self.__init__(self.event_list[0],self.event_list[1],self.event_list[2],self.event_list[3],self.event_list[4],self.event_list[5],self.event_list[6],self.event_list[7])
