#include<bits/stdc++.h>
using namespace std;
long long int bigmod(long long int g, long long int a, long long int p){
    long long int i,res=1;
    for(i=1;i<=a;i++){
        res = res * g;
        res = res % p;
    }
    return res;
}
int main()
{
   long long int p,g,a,b,x,y,xa,ya,ka,kb;
    cout<<"enter the number of p and g\n";
    cin>>p>>g;
    cout<<"enter the number of a and b\n";
    cin>>a>>b;
    x = bigmod(g,a,p);
    y = bigmod(g,b,p);
    ka = bigmod(y,a,p);
    kb = bigmod(x,b,p);
    cout<<ka<<" "<<kb<<endl;
    return 0;

}
