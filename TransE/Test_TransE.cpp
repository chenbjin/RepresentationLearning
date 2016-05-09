/*
  Code is modified from https://github.com/Mrlyk423/Relation_Extraction
  @chenbingjin 2016-05-05
*/
#include <iostream>
#include <cstring>
#include <cstdio>
#include <map>
#include <vector>
#include <string>
#include <ctime>
#include <algorithm>
#include <cmath>
#include <cstdlib>
using namespace std;

/*
    Evaluation for link prediction used in (Bordes, 2013)
    Method: Mean Rank and Hits@10 (Bordes,2011, aaai)
    input:
	entity2vec,relation2vec,test datasets, train datasets
    output:
	mean rank: 平均rank越低越好
        Hits@10(%): 前十命中越高越好
    备注：考虑测试效率，可以减少测试集的大小。4w 测试集要跑2个半小时，当然可以用多线程改进。
*/

bool debug=false;
bool L1_flag=1;

string version;
string trainortest = "test";

map<string,int> relation2id,entity2id;
map<int,string> id2entity,id2relation;

map<int,map<int,int> > entity2num;  //某个关系下实体出现的次数
map<int,int> e2num;  // 实体出现次数

int relation_num,entity_num;
int n= 50;

double sigmod(double x)
{
    return 1.0/(1+exp(-x));
}

double vec_len(vector<double> a)
{
    double res=0;
    for (int i=0; i<a.size(); i++)
        res+=a[i]*a[i];
    return sqrt(res);
}
// 输出向量
void vec_output(vector<double> a)
{
    for (int i=0; i<a.size(); i++)
    {
        cout<<a[i]<<"\t";
        if (i%10==9)
            cout<<endl;
    }
    cout<<"-------------------------"<<endl;
}

double sqr(double x)
{
    return x*x;
}

char buf[100000],buf1[100000], buf2[10000], buf3[10000];

int my_cmp(pair<double,int> a,pair<double,int> b)
{
    return a.first>b.first;
}

double cmp(pair<int,double> a, pair<int,double> b)
{
    return a.second<b.second;
}

class Test{
    vector<vector<double> > relation_vec,entity_vec;

    vector<int> h,l,r;
    vector<int> fb_h,fb_l,fb_r;
    map<pair<int,int>, map<int,int> > ok;
    double res ;
public:
    // 添加测试集
    void add(int x,int y,int z, bool flag)
    {
        if (flag)
        {
            fb_h.push_back(x);
            fb_r.push_back(z);
            fb_l.push_back(y);
        }
        ok[make_pair(x,z)][y]=1;
    }

    int rand_max(int x)
    {
        int res = (rand()*rand())%x;
        if (res<0)
            res+=x;
        return res;
    }
    double len;
    double calc_sum(int e1,int e2,int rel)
    {
        double sum=0;
        if (L1_flag)
            for (int ii=0; ii<n; ii++)
                sum+=-fabs(entity_vec[e2][ii]-entity_vec[e1][ii]-relation_vec[rel][ii]);
        else
            for (int ii=0; ii<n; ii++)
                sum+=-sqr(entity_vec[e2][ii]-entity_vec[e1][ii]-relation_vec[rel][ii]);
        return sum;
    }
    // 测试过程
    void run()
    {
        // 导入实体向量，关系向量
        FILE* f1 = fopen(("./vec/relation2vec."+version).c_str(),"r");
        FILE* f3 = fopen(("./vec/entity2vec."+version).c_str(),"r");
        cout << "relation num: " << relation_num << " , entity num: "<<entity_num<<endl;
        int relation_num_fb=relation_num;
        relation_vec.resize(relation_num_fb);
        for (int i=0; i<relation_num_fb;i++)
        {
            relation_vec[i].resize(n);
            for (int ii=0; ii<n; ii++)
                fscanf(f1,"%lf",&relation_vec[i][ii]);
        }
        entity_vec.resize(entity_num);
        for (int i=0; i<entity_num;i++)
        {
            entity_vec[i].resize(n);
            for (int ii=0; ii<n; ii++)
                fscanf(f3,"%lf",&entity_vec[i][ii]);
            if (vec_len(entity_vec[i])-1>1e-3)
                cout<<"wrong_entity"<<i<<' '<<vec_len(entity_vec[i])<<endl;
        }
        fclose(f1);
        fclose(f3);

        double lsum=0 ,lsum_filter= 0;  //记录左边实体的rank，以及filter后的rank
        double rsum = 0,rsum_filter=0;
        double lp_n=0,lp_n_filter;  //top10
        double rp_n=0,rp_n_filter;
        map<int,double> lsum_r,lsum_filter_r;  //记录关系对应实体的rank
        map<int,double> rsum_r,rsum_filter_r;
        map<int,double> lp_n_r,lp_n_filter_r;
        map<int,double> rp_n_r,rp_n_filter_r;
        map<int,int> rel_num;  // 测试集关系出现次数

        cout << "Test triplets num: " << fb_l.size() << endl; 
        for (int testid = 0; testid<fb_l.size(); testid+=1)
        {
            cout << testid << endl;
            int h = fb_h[testid];
            int l = fb_l[testid];
            int rel = fb_r[testid];
            double tmp = calc_sum(h,l,rel);
            rel_num[rel]+=1;
            // （h，rel，l）替换h后，计算每个实体的非相似性 （Bordes，2011）
            vector<pair<int,double> > a; 
            for (int i=0; i<entity_num; i++)
            {
                double sum = calc_sum(i,l,rel);
                a.push_back(make_pair(i,sum));
            }
            sort(a.begin(),a.end(),cmp);
            double ttt=0;
            int filter = 0;
            for (int i=a.size()-1; i>=0; i--)
            {
                if (ok[make_pair(a[i].first,rel)].count(l)>0)
                    ttt++;
                if (ok[make_pair(a[i].first,rel)].count(l)==0)
                    filter+=1;
                if (a[i].first ==h)  //记录正确实体的rank
                {
                    //cout <<"hit: " << i << endl; 
                    lsum+=a.size()-i;   
                    lsum_filter+=filter+1;
                    lsum_r[rel]+=a.size()-i;
                    lsum_filter_r[rel]+=filter+1;
                    if (a.size()-i<=10)
                    {
                        lp_n+=1;
                        lp_n_r[rel]+=1;
                    }
                    if (filter<10)
                    {
                        lp_n_filter+=1;
                        lp_n_filter_r[rel]+=1;
                    }
                    break;
                }
            }
            /* 注：为了方便，我只测试了替换h的效果
            // （h，rel，l）替换l后，计算每个实体的非相似性 （Bordes，2011）
            a.clear();
            for (int i=0; i<entity_num; i++)
            {
                double sum = calc_sum(h,i,rel);
                a.push_back(make_pair(i,sum));
            }
            sort(a.begin(),a.end(),cmp);
            ttt=0;
            filter=0;
            for (int i=a.size()-1; i>=0; i--)
            {
                if (ok[make_pair(h,rel)].count(a[i].first)>0)
                    ttt++;
                if (ok[make_pair(h,rel)].count(a[i].first)==0)
                    filter+=1;
                if (a[i].first==l)
                {
                    rsum+=a.size()-i;
                    rsum_filter+=filter+1;
                    rsum_r[rel]+=a.size()-i;
                    rsum_filter_r[rel]+=filter+1;
                    if (a.size()-i<=10)
                    {
                        rp_n+=1;
                        rp_n_r[rel]+=1;
                    }
                    if (filter<10)
                    {
                        rp_n_filter+=1;
                        rp_n_filter_r[rel]+=1;
                    }
                    break;
                }
            }*/
        }
        cout<<"---------left----------"<<endl;
        cout<<"Mean Rank: "<<lsum/fb_l.size() << "(Raw)"<<'\t'<<lsum_filter/fb_l.size()<<"(Filter)"<<endl;
        cout<<"Hits@10(%): "<<lp_n/fb_l.size() << "(Raw)"<<'\t'<<lp_n_filter/fb_l.size()<<"(Filter)"<<endl;
	/*
        cout<<"--------right----------"<<endl;
        cout<<"Mean Rank: "<<rsum/fb_r.size() << "(Raw)"<<'\t'<<rsum_filter/fb_r.size()<<"(Filter)"<<endl;
        cout<<"Hits@10(%): "<<rp_n/fb_r.size() << "(Raw)"<<'\t'<<rp_n_filter/fb_r.size()<<"(Filter)"<<endl;*/
    }
};
Test test;

// 数据准备
void prepare()
{
    cout << "loading entity2id, relation2id ..." << endl;
    FILE* f1 = fopen("./data/entity2id.txt","r");
    FILE* f2 = fopen("./data/relation2id.txt","r");
    int x;
    // 数据处理过程，按行读取，然后再格式化分割（要考虑实体名字含有空格）
    while (!feof(f1))
    {
        fgets(buf1,4096,f1);  // 获取一行
        sscanf(buf1,"%[^\t]\t%d\n", buf, &x);
        string st=buf;
        entity2id[st]=x;
        id2entity[x]=st;
        mid2type[st]="None";
        entity_num++;
    }
    while (!feof(f2))
    {
        fgets(buf1,4096,f2);	
        sscanf(buf1,"%[^\t]\t%d\n", buf, &x);
        string st=buf;
        relation2id[st]=x;
        id2relation[x]=st;
        relation_num++;
    }
    fclose(f1);
    fclose(f2);
    int c;
    /* 考虑中文数据读取问题（实体名可能有空格），按行读取，再格式化切割*/
    cout << "loading test datasets ..." << endl;
    FILE* f_kb = fopen("./data/test.txt","r");
    while (!feof(f_kb))
    {
        fgets(buf, 20480, f_kb);
        sscanf(buf, "%[^\t]\t%[^\t]\t%[^\t\n]\n", buf1, buf2, buf3);	
        string s1=buf1;
        string s2=buf3;
        string s3=buf2;
        if (entity2id.count(s1)==0)
        {
            cout<<"miss entity:"<<s1<<endl;
            c=getchar();
        }
        if (entity2id.count(s2)==0)
        {
            cout<<"miss entity:"<<s2<<endl;
        }
        if (relation2id.count(s3)==0)
        {
            cout<<"miss relation:"<<s3<<endl;
            relation2id[s3] = relation_num;
            relation_num++;
        }
        test.add(entity2id[s1],entity2id[s2],relation2id[s3],true);
    }
    fclose(f_kb);

    cout << "loading train datasets ..." << endl;
    FILE* f_kb1 = fopen("./data/train.txt","r");
    while (!feof(f_kb1))
    {
        fgets(buf, 20480, f_kb1);
        sscanf(buf, "%[^\t]\t%[^\t]\t%[^\t\n]\n", buf1, buf2, buf3);	
        string s1=buf1;
        string s2=buf3;
        string s3=buf2;
        if (entity2id.count(s1)==0)
        {
            cout<<"miss entity:"<<s1<<endl;
        }
        if (entity2id.count(s2)==0)
        {
            cout<<"miss entity:"<<s2<<endl;
        }
        if (relation2id.count(s3)==0)
        {
            relation2id[s3] = relation_num;
            relation_num++;
        }

        entity2num[relation2id[s3]][entity2id[s1]]+=1;
        entity2num[relation2id[s3]][entity2id[s2]]+=1;
        e2num[entity2id[s1]]+=1;
        e2num[entity2id[s2]]+=1;
        test.add(entity2id[s1],entity2id[s2],relation2id[s3],false);
    }
    fclose(f_kb1);

    cout << "loading vaid datasets ..." << endl;
    FILE* f_kb2 = fopen("./data/valid.txt","r");
    while (!feof(f_kb2))
    {
        fgets(buf, 20480, f_kb2);	
        sscanf(buf, "%[^\t]\t%[^\t]\t%[^\t\n]\n", buf1, buf2, buf3);	
        string s1=buf1;
        string s2=buf3;
        string s3=buf2;
        if (entity2id.count(s1)==0)
        {
            cout<<"miss entity:"<<s1<<endl;
        }
        if (entity2id.count(s2)==0)
        {
            cout<<"miss entity:"<<s2<<endl;
        }
        if (relation2id.count(s3)==0)
        {
            relation2id[s3] = relation_num;
            relation_num++;
        }
        test.add(entity2id[s1],entity2id[s2],relation2id[s3],false);
    }
    fclose(f_kb2);
}


int main(int argc,char**argv)
{
    if (argc<2)
        return 0;
    else
    {
        clock_t start_time = clock();
        version = argv[1];
        prepare();
        test.run();
        clock_t end_time = clock();
        printf("cost time:%.2f mins.\n ",static_cast<double>(end_time-start_time)/CLOCKS_PER_SEC*1000*1000*60);
    }
}

