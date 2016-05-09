/*
  Code is modified from https://github.com/Mrlyk423/Relation_Extraction
  @chenbingjin 2016-05-03
*/
#include <iostream>
#include <cstring>
#include <cstdio>
#include <map>
#include <vector>
#include <string>
#include <ctime>
#include <cmath>
#include <cstdlib>
#include <time.h>
using namespace std;

#define pi 3.1415926535897932384626433832795

bool L1_flag = 0; //默认采用L1=1

// 随机数
double rand(double min, double max)
{
    return min + (max-min)*rand()/(RAND_MAX + 1.0);
}
// 正态分布
double normal(double x, double miu,double sigma)
{
    return 1.0/sqrt(2*pi)/sigma*exp(-1*(x-miu)*(x-miu)/(2*sigma*sigma));
}
// 在[min,max]区间内做正态分布采样？
double randn(double miu,double sigma, double min ,double max)
{
    double x,y,dScope;
    do{
        x=rand(min,max);
        y=normal(x,miu,sigma);
        dScope=rand(0.0,normal(miu,miu,sigma));
    }while(dScope>y);
    return x;
}
// 平方
double sqr(double x)
{
    return x*x;
}
// 向量的模（L2）
double vec_len(vector<double> &a)
{
    double res=0;
    for (int i=0; i<a.size(); i++)
        res+=a[i]*a[i];
    res = sqrt(res);
    return res;
}

string version;
/* 构造负样本的采样方法 (Wang,2014)
    version = "bern";  对于1-N，N-1的关系，以较大的概率替换1的实体
    version = "unif";  传统方法：随机替换
*/

char buf[100000],buf1[100000],buf2[100000];
int relation_num,entity_num;
//关系id映射，实体id映射
map<string,int> relation2id,entity2id; 
map<int,string> id2entity,id2relation; 

/*
    记录head实体和tail实体关系类型：1-1,1-N，N-1，N-N
    对于三元组（h,r,t）
    left_entity[r][h] ++
    right_entity[r][t] ++
*/
map<int,map<int,int> > left_entity,right_entity;
map<int,double> left_num,right_num;

class Train{

public:
    map<pair<int,int>, map<int,int> > ok;
    // 添加三元组训练集
    void add(int x,int y,int z)
    {
        fb_h.push_back(x);
        fb_r.push_back(z);
        fb_l.push_back(y);
        ok[make_pair(x,z)][y]=1;
    }
    // transE算法学习过程（Bordes，2013）
    void run(int n_in,double rate_in,double margin_in,int method_in)
    {
        cout << "Initing vector..." << endl;
        n = n_in;
        rate = rate_in;
        margin = margin_in;
        method = method_in;
        // 申请对应的向量空间
        relation_vec.resize(relation_num);
        for (int i=0; i<relation_vec.size(); i++)
            relation_vec[i].resize(n);
            entity_vec.resize(entity_num);
        for (int i=0; i<entity_vec.size(); i++)
            entity_vec[i].resize(n);
            relation_tmp.resize(relation_num);
        for (int i=0; i<relation_tmp.size(); i++)
            relation_tmp[i].resize(n);
            entity_tmp.resize(entity_num);
        for (int i=0; i<entity_tmp.size(); i++)
            entity_tmp[i].resize(n);

        cout << "relation vector initing..." << endl;
        // 关系向量初始化
        for (int i=0; i<relation_num; i++)
        {
            for (int ii=0; ii<n; ii++)
                relation_vec[i][ii] = randn(0,1.0/n,-6/sqrt(n),6/sqrt(n));
        }
        // 实体向量初始化
        cout << "entity vector initing..." << endl;
        for (int i=0; i<entity_num; i++)
        {
            for (int ii=0; ii<n; ii++)
                entity_vec[i][ii] = randn(0,1.0/n,-6/sqrt(n),6/sqrt(n));
            norm(entity_vec[i]); //归一化
        }
        // BFGS优化求解
        cout << "BFGS ..." << endl;
        bfgs();
    }

private:
    int n,method; // n为向量的维度
    double res;   // 合页损失
    double count,count1;// 损失函数梯度
    double rate,margin; 
    vector<int> fb_h,fb_l,fb_r; //(h,r,l)
    vector<vector<double> > relation_vec,entity_vec; // 关系向量，实体向量
    vector<vector<double> > relation_tmp,entity_tmp; // 优化求解过程，临时向量
    // 归一化
    double norm(vector<double> &a)
    {
        double x = vec_len(a);
        if (x>1)
        for (int ii=0; ii<a.size(); ii++)
            a[ii]/=x;
        return 0;
    }
    // 随机数
    int rand_max(int x)
    {
        int res = (rand()*rand())%x;
        while (res<0)
            res+=x;
        return res;
    }
    // BFGS algorithm，update embeddings
    void bfgs()
    {
        res=0;
        int nbatches=100;
        int nepoch = 300;  //100-1000
        int batchsize = fb_h.size()/nbatches; //每个batch的size
        cout << "batchsize: " << batchsize << endl;
        
        for (int epoch=0; epoch<nepoch; epoch++)
        {
            res=0;
            clock_t start_time = clock();
            for (int batch = 0; batch<nbatches; batch++)
            {
                relation_tmp=relation_vec;
                entity_tmp = entity_vec;
                // sample and train a minibatch of size batchsize
                for (int k=0; k<batchsize; k++)
                {
                    // 构造负样本的采样方法 (Wang,2014)
                    int i=rand_max(fb_h.size());
                    int j=rand_max(entity_num);
                    double pr = 1000*right_num[fb_r[i]]/(right_num[fb_r[i]]+left_num[fb_r[i]]);
                    if (method ==0)
                        pr = 500;
                    if (rand()%1000<pr)
                    {
                        while (ok[make_pair(fb_h[i],fb_r[i])].count(j)>0)
                            j=rand_max(entity_num);
                        // 训练正负样本（h,r,l）（h,r,l'）
                        train_kb(fb_h[i],fb_l[i],fb_r[i],fb_h[i],j,fb_r[i]);
                    }
                    else
                    {
                        while (ok[make_pair(j,fb_r[i])].count(fb_l[i])>0)
                            j=rand_max(entity_num);
                        // 训练正负样本（h,r,l）（h',r,l）
                        train_kb(fb_h[i],fb_l[i],fb_r[i],j,fb_l[i],fb_r[i]);
                    }
                    norm(relation_tmp[fb_r[i]]);
                    norm(entity_tmp[fb_h[i]]);
                    norm(entity_tmp[fb_l[i]]);
                    norm(entity_tmp[j]);
                }
                relation_vec = relation_tmp;
                entity_vec = entity_tmp;
            }
            clock_t end_time = clock();
            cout<<"epoch: "<<epoch<<", traning time: "<< static_cast<double>(end_time-start_time)/CLOCKS_PER_SEC*1000 <<"ms, res:" <<res<<endl;

            // 保存向量
            FILE* f2 = fopen(("./vec/relation2vec."+version).c_str(),"w");
            FILE* f3 = fopen(("./vec/entity2vec."+version).c_str(),"w");
            for (int i=0; i<relation_num; i++)
            {
                for (int ii=0; ii<n; ii++)
                    fprintf(f2,"%.6lf\t",relation_vec[i][ii]);
                fprintf(f2,"\n");
            }
            for (int i=0; i<entity_num; i++)
            {
                for (int ii=0; ii<n; ii++)
                    fprintf(f3,"%.6lf\t",entity_vec[i][ii]);
                fprintf(f3,"\n");
            }
            fclose(f2);
            fclose(f3);
        }
    }
    double res1;
    // 计算（l-h-r)的损失，累加向量的每维
    double calc_sum(int e1,int e2,int rel)
    {
        double sum=0;
        if (L1_flag)
            for (int ii=0; ii<n; ii++)
                sum+=fabs(entity_vec[e2][ii]-entity_vec[e1][ii]-relation_vec[rel][ii]);
        else
            for (int ii=0; ii<n; ii++)
                sum+=sqr(entity_vec[e2][ii]-entity_vec[e1][ii]-relation_vec[rel][ii]);
        return sum;
    }
    // 计算梯度，更新向量
    void gradient(int e1_a,int e2_a,int rel_a,int e1_b,int e2_b,int rel_b)
    {
        for (int ii=0; ii<n; ii++)
        {
            double x = 2*(entity_vec[e2_a][ii]-entity_vec[e1_a][ii]-relation_vec[rel_a][ii]);
            if (L1_flag)
                if (x>0)
                    x=1;
                else
                    x=-1;
            relation_tmp[rel_a][ii]-=-1*rate*x;
            entity_tmp[e1_a][ii]-=-1*rate*x;
            entity_tmp[e2_a][ii]+=-1*rate*x;
            x = 2*(entity_vec[e2_b][ii]-entity_vec[e1_b][ii]-relation_vec[rel_b][ii]);
            if (L1_flag)
                if (x>0)
                    x=1;
                else
                    x=-1;
            relation_tmp[rel_b][ii]-=rate*x;
            entity_tmp[e1_b][ii]-=rate*x;
            entity_tmp[e2_b][ii]+=rate*x;
        }
    }
    // 训练过程：计算损失（L1或L2），计算梯度，更新向量
    void train_kb(int e1_a,int e2_a,int rel_a,int e1_b,int e2_b,int rel_b)
    {
        double sum1 = calc_sum(e1_a,e2_a,rel_a);
        double sum2 = calc_sum(e1_b,e2_b,rel_b);
        if (sum1+margin>sum2)
        {
            res+=margin+sum1-sum2;
            gradient( e1_a, e2_a, rel_a, e1_b, e2_b, rel_b);
        }
    }
};

Train train;
// 训练数据准备
void prepare()
{
    /* 
    需考虑实体中可能含有空格（eg.'psp go'），故采用按行读取，再划分实体名/关系名和id
    */
    FILE* f1 = fopen("./data/entity2id.txt","r");
    FILE* f2 = fopen("./data/relation2id.txt","r");
    int x;
    cout << "Reading entity2id ..." << endl;
    while (!feof(f1))
    {
        fgets(buf1,4096,f1);
        sscanf(buf1,"%[^\t]\t%d\n",buf,&x);
        string st=buf;
        entity2id[st]=x;
        id2entity[x]=st;
        entity_num++;
        if (x % 500000 == 0)
            cout << st << " " << x << endl;
    }
    cout << entity2id.size() << endl;
    cout << "Reading relation2id ..." << endl;
    while (!feof(f2))
    {	
        fgets(buf1,4096,f2);
        sscanf(buf1,"%[^\t]\t%d\n",buf,&x);
        string st=buf;
        relation2id[st]=x;
        id2relation[x]=st;
        relation_num++;
    }
    //cout << "Press num to read training data" << endl;
    //int c;
    //c = getchar();
    FILE* f_kb = fopen("./data/train.txt","r");
    char buf3[40960];
    cout << "Loading training data..." << endl;
    while (!feof(f_kb))
    {
        fgets(buf,20480,f_kb);
        sscanf(buf,"%[^\t]\t%[^\t]\t%[^\t\n]\n", buf1,buf2,buf3);
        string s1=buf1;
        string s2=buf3;
        string s3=buf2; //relation
        //cout << s1 << " " << s3 << " " << s2 << endl;
        if (entity2id.count(s1)==0)
        {
            cout<<"miss head entity:"<<s1<<endl;
            // c=getchar();
        }
        if (entity2id.count(s2)==0)
        {
            cout<<"miss tail entity:"<<s2<<endl;
            // c=getchar();
        }
        if (relation2id.count(s3)==0)
        {
            relation2id[s3] = relation_num;
            relation_num++;
        }
        left_entity[relation2id[s3]][entity2id[s1]]++;
        right_entity[relation2id[s3]][entity2id[s2]]++;
        train.add(entity2id[s1],entity2id[s2],relation2id[s3]);
    }
    // 计算每个关系head实体出现的平均数
    for (int i=0; i<relation_num; i++)
    {
        double sum1=0,sum2=0;
        for (map<int,int>::iterator it = left_entity[i].begin(); it!=left_entity[i].end(); it++)
        {
            sum1++;
            sum2+=it->second;
        }
        left_num[i]=sum2/sum1;
    }
    // 计算每个关系tail实体出现的平均数
    for (int i=0; i<relation_num; i++)
    {
        double sum1=0,sum2=0;
        for (map<int,int>::iterator it = right_entity[i].begin(); it!=right_entity[i].end(); it++)
        {
            sum1++;
            sum2+=it->second;
        }
        right_num[i]=sum2/sum1;
    }
    cout<<"relation_num="<<relation_num<<endl;
    cout<<"entity_num="<<entity_num<<endl;
    fclose(f_kb);
}
// 参数匹配
int ArgPos(char *str, int argc, char **argv) {
    int a;
    for (a = 1; a < argc; a++)
        if (!strcmp(str, argv[a])) {
            if (a == argc - 1) {
                printf("Argument missing for %s\n", str);
                exit(1);
            }
            return a;
        }
    return -1;
}

/*
main函数
Train_TransE [-size 100][-rate 0.01][-method 1]
para:
    -size: embedding size n, default: n = 100
    -method: 0-unif,1-bern, default: method = 1
    -rate: learning rate, default: rate = 0.001
*/
int main(int argc,char**argv)
{
    srand((unsigned) time(NULL));
    int method = 1;
    int n = 50;
    double rate = 0.1;
    double margin = 1;
    int i;
    if ((i = ArgPos((char *)"-size", argc, argv)) > 0) n = atoi(argv[i + 1]);
    //if ((i = ArgPos((char *)"-margin", argc, argv)) > 0) margin = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-rate", argc, argv)) > 0) rate = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-method", argc, argv)) > 0) method = atoi(argv[i + 1]);
    cout<<"size = "<<n<<endl;
    cout<<"learing rate = "<<rate<<endl;
    cout<<"margin = "<<margin<<endl;
    if (method)
        version = "bern";   // 采样方法
    else
        version = "unif";
    cout<<"method = "<<version<<endl;
    prepare();
    train.run(n,rate,margin,method);
}


