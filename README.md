# AI_model

画像を識別するAIのひな型になります。  
改変等は引用明記の上ご自由にお使いください 。 


ディレクトリの設置の仕方は以下のようにすれば動きます
Image,Train,Testはディレクトリです。パスは適宜変更してください  
/root  
|__AI_learn.py  
|  
|__AI_test.py  
|  
|__Image__  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|__Train  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|__Test  
          
精度を上げたいときは  
・model.compile()の引数を変える  
・CNNの層を変える  
・データを増やす  
とすると精度がよくなる可能性があります。
