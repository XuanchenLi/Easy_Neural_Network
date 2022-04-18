import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import NNF as nf

class BP_neueal_network:
    def __init__(self,X,Y,n_input,n_output,n_h,fuction="ReLU",loss="CrossEntropy",optimizer="AdaGrad",h_l_num=1,learning_rate=0.1,iterations=50,batch_size=20):
        #输入数据，输出标签，输入层神经元个数，输出层神经元个数，隐含层神经元个数，激活函数，损失函数，优化器，隐含层层数，学习率，训练次数，批大小

        self.features=X#原始输入特征
        self.input_num=n_input#输入数据维数
        self.output_num=n_output#输出数据维数
        self.hidden=n_h#隐藏层神经元个数
        self.fuction=fuction#激励函数
        self.loss=loss
        self.optimizer=optimizer#优化器
        self.hidden_layer_num=h_l_num#隐含层数
        self.learning_rate=learning_rate#学习率
        self.epoch=iterations#迭代次数
        self.batch_size=batch_size#批数量
        self.input_node=nf.core.Variable(dim=(self.input_num, 1), init=False, trainable=False)#计算图结点
        self.output_node=nf.core.Variable(dim=(self.output_num, 1), init=False, trainable=False)#计算图结点
        self.ini_lable=Y#原始标签
        #one_hot编码后标签
        if(self.output_num>1):
            le = LabelEncoder()
            number_label = le.fit_transform(Y)
            # 将整数形式的标签转换成One-Hot编码
            oh = OneHotEncoder(sparse=False)
            self.lable=oh.fit_transform(number_label.reshape(-1, 1))
        else:
            self.lable=Y


    def layerbuild(self):
        """
         搭建隐藏层和输出层
         返回输出层

         """
        formarlayer=nf.layer.fc(self.input_node,self.input_num,self.hidden,self.fuction)
        hidden_layer_num=self.hidden_layer_num
        #隐藏层
        while(hidden_layer_num>1):
            newlayer=nf.layer.fc(formarlayer,self.hidden,self.hidden,self.fuction)
            formarlayer=newlayer
            hidden_layer_num=hidden_layer_num-1
        #输出层
        outputlayer=nf.layer.fc(formarlayer,self.hidden,self.output_num,None)
        return outputlayer

    def lossfunctionset(self,outputlayer):
        """
        损失函数选择
        """
        if self.loss=="CrossEntropy":
            return  nf.operators.loss.CrossEntropyWithSoftMax(outputlayer, self.output_node)
        elif self.loss=="LogLoss":
            return nf.operators.loss.LogLoss(outputlayer,self.output_node)

    def optimizerset(self,loss):
        """
        优化器选择
        """
        if(self.optimizer=="AdaGrad"):
            return nf.optimizer.Adam(nf.default_graph,loss,self.learning_rate)
        elif(self.optimizer=="RMSProp"):
            return nf.optimizer.RMSProp(nf.default_graph,loss,self.learning_rate)
        elif(self.optimizer=="Momentum"):
            return nf.optimizer.Momentum(nf.default_graph,loss,self.learning_rate)

    def network_evaluation(self,predict_node):
        """
        正确性评价
        """
        #结果存储矩阵
        pred=[]
        for item in range(len(self.features)):
            feature=np.mat(self.features[item,:]).T
            self.input_node.set_value(feature)
            #前向传播
            predict_node.forward()
            pred.append(predict_node.value.A.ravel())
        #多分类问题结果降维
        if(self.output_num>1):pred=np.array(pred).argmax(axis=1)
        return (self.ini_lable == pred).astype(np.int).sum()/len(self.features)

    def train(self):
        """
        神经网络训练主函数
        """
        #隐含层输出层搭建
        outputlayer=self.layerbuild()
        #加入预测结点
        predict_node=nf.operators.SoftMax(outputlayer)
        #定义损失函数
        loss=self.lossfunctionset(outputlayer);
        #定义优化器
        optimizer=self.optimizerset(loss)
        #开始训练
        for epoch in range(self.epoch):
            # 批计数器
            batch_count=0
            #遍历样本
            for item in range(len(self.features)):
                #item个训练样本的特征和类标签
                feature=np.mat(self.features[item,:]).T
                lable=np.mat(self.lable[item,:]).T
                #将特征赋值给计算图的结点
                self.input_node.set_value(feature)
                self.output_node.set_value(lable)
                #调用优化器先进行前向再进行反向传播
                optimizer.step()
                #计数器自增
                batch_count+=1
                if batch_count>self.batch_size:
                    optimizer.update()
                    batch_count=0
            #模型评价
            acc=self.network_evaluation(predict_node)
            print("epoch: {:d}, accuracy: {:.3f}".format(epoch + 1, acc))



data = pd.read_csv(".\iris_training.csv")
# 随机打乱样本顺序
data = data.sample(len(data), replace=False)
features = data[['SepalLength',
 'SepalWidth',
 'PetalLength',
 'PetalWidth']].values
lable=data["species"]
network=BP_neueal_network(features,lable,4,3,10,h_l_num=3)
network.train()