
# Basic code optimization techniques



## SIMD instrinsics

http://kristofferc.github.io/post/intrinsics/ 



## Loop unrolling

```
void vsum1(int n) {
  int i;
  for(i=0; i<n; i++)
    c[i] = a[i] + b[i];
}
```

Can be rewritten as

```
void vsum2(int n) {
  int i;
  for(i=0; i<n; i+=2) {
    c[i] = a[i] + b[i];
    c[i+1] = a[i+1] + b[i+1];
}
```



### Matrix multiply blog

https://gist.github.com/nadavrot/5b35d44e8ba3dd718e595e40184d03f0