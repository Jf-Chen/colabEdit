5-way 5-shot下的train过程
train_shot=5 # 
train_way=5
train_query_shot=15
query_way=5

self.psi = nn.Sequential(
            # nn.Linear() y=x*A'+b，A是weight，
            nn.Linear(input_dimension, input_dimension  * 2), 
            nn.ReLU(),
            nn.Linear(input_dimension * 2, input_dimension * 2),
            nn.ReLU()
        )
self.rho = nn.Sequential(
            nn.Linear(input_dimension * 3, input_dimension * 2),
            nn.ReLU(),
            nn.Linear(input_dimension * 2, output_dimension),
        )


glofa:

feature_map=[100, 25, 640] # [train_way*(train_shot+train_query_shot),resolution,channels]
d=640
channels=640

init()
self.f_task = SetFunction(args, input_dimension=dimension, output_dimension=dimension)
self.f_class = SetFunction(args, input_dimension=dimension, output_dimension=dimension)
self.h = SetFunction(args, input_dimension=dimension, output_dimension=2)

forward()

support_embeddings = feature_map[:train_way*train_shot] # [25,25,640]
query_embeddings = embeddings[train_way*train_shot:] # [75,25,640]
#------ mask_task  --------#
mask_task = self.f_task(support_embeddings, level='task')
    
f_task(support_embeddings,level='task')
        psi_output = self.psi(support_embeddings) #[25, 25, 1280]
        rho_input = torch.cat([psi_output, support_embeddings], dim=2) # [25, 25, 1920]
        rho_input = torch.sum(rho_input, dim=0, keepdim=True) # [1, 25, 1920]
        #--------暂时去掉delta--------#
        # rho_output = F.relu6(self.rho(rho_input)) / 6 * self.args.delta
        rho_output = torch.nn.functional.relu6(self.rho(rho_input)) / 6 #[1, 25, 640]
        #-----------end-------------# 

#-------mask_class-----------#
mask_class=selft.f_class(support_embeddings,level='class')  

mask_class = self.f_class(support_embeddings, level='class')
        psi_output = self.psi(support_embeddings) # [25, 25, 1280]
        rho_input = torch.cat([psi_output, support_embeddings], dim=2)# [25, 25, 1920]
        rho_input = rho_input.view(train_way, train_shot,resolution, -1) # [5, 5, 25, 1920]
        rho_input = torch.sum(rho_input, dim=1) # [5, 25, 1920] # 含义是每个类的mask
        # rho_output = torch.nn.functional.relu6(self.rho(rho_input)) / 6 * self.args.delta
        rho_output = torch.nn.functional.relu6(f_class.rho(rho_input)) / 6 # [5, 25, 640]
        rho_output = rho_output.unsqueeze(0) # [1, 5, 25, 640]
        return rho_output

masked_support_embeddings = support.view(train_way,train_shot, resolution,-1) * \
            (1 + mask_task ) * (1 + mask_class )# [5, 5, 25, 640]
            

          
masked_query_embeddings =query.unsqueeze(0).expand(train_way, -1, -1,-1) * \
            (1 + mask_task) * (1 + mask_class.transpose(0, 1) ) # [5, 75, 25, 640]