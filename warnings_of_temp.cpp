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
    void show()
    {
        cout<<a<<"   "<<b<<endl;
    }
};
template <typename T1>
T1 add(T1& t)
{
    return ++t;
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
int main()
{
    system("chcp 65001 > nul");  // 加上这行，强制切换控制台为 UTF-8
    double gg=7.0;
    girl g(gg);
    swa(g);
    g.show();




    return 0;
}