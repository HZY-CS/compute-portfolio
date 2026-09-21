#include <iostream>
#include <vector>
#include <cstdlib>  
using namespace std;
//为类函数 创建模板时 是不能给 虚函数和析构加模板
template<typename T>
class girl
{
    int a,b;
    public:
    girl(T& t)
    {
        cout<<"girl:"<<t<<endl;
    }
    template <typename T1> friend void swa(girl<T1>& g);
    template<typename T1> friend void swap1(girl<T1> &g1,girl<T1> &g2);
    void show()
    {
        cout<<a<<"   "<<b<<endl;
    }
};
template <typename T1>
T1 add(T1& t)
{
    cout<<"普通模板"<<endl;
    return ++t;
}
template<>             //具体化
int add<int>(int& a)
{
    cout<<"具体化模板"<<endl;
    return ++a;
}
int add(int& a)
{
    cout<<"普通函数"<<endl;
    return ++a;
}
template <typename T,typename a,typename b>
T add(T& t,a& a1,b& b1)      //也可以用模板 去重载
{
    cout<<t<<"   "<<a1+b1<<endl;
    return t;
}

template <typename T> void swa(girl<T>&);
template<typename T>
void swa(girl<T>& g)
{      
    g.a++;
}
template <typename T>
void swap1(T& t1,T& t2)
{
    T t=t1;
    t1=t2;
    t2=t;
}
// 特化：类型必须写死（全特化），不能留模板参数（部分特化）。
// 重载：同名函数，接受更具体的类型（如 girl<T>），编译器会优先匹配更具体的版本。
// 普通函数(重载)>具体化模板>普通模板 

template<typename T1>
void swap1(girl<T1> &g1,girl<T1> &g2)
{
    swap1(g1.a,g2.a);
    swap1(g1.b,g2.b);
}
int main()
{
    system("chcp 65001 > nul");  // 加上这行，强制切换控制台为 UTF-8
    double gg=7.0;
    girl g(gg);
    swa(g);
    g.show();
    girl g2(gg);
    swap1(g,g2);    //交换
    g2.show();
    g.show();
    int a=8;
    add(a);
    add<>(a);
    add(gg);





    return 0;
}