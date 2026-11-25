// Host decode for dettato's grouper. Port of dettato 2.0.0, Apache-2.0.
#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <vector>

namespace dettato {
using Vec = std::vector<double>;
using Matrix = std::vector<Vec>;
struct Window { double start; std::vector<int> indices; Matrix embedding, probability; };
struct Prediction { double window_seconds; Matrix embedding, probability, hidden; std::vector<Window> windows; };
struct Note { double onset, offset, pitch; };
struct Layer { Matrix weights; Vec bias; };
struct Group { int instrument; std::vector<int> notes; };
struct DecodeOptions {
    double tau = .1, tau2 = .4, lambda = .5, alpha = .5, absorb = .3, absorb_fraction = .25;
    int min_notes = 30, refine = 2;
    bool expected_credit = true, merge = true;
};

inline double dot(const Vec &a, const Vec &b) {
    double s = 0; for (size_t i = 0; i < a.size(); ++i) s += a[i] * b[i]; return s;
}
inline Vec unit(Vec a) {
    double n = std::max(std::sqrt(dot(a, a)), 1e-9); for (double &v : a) v /= n; return a;
}
inline std::vector<int> unique_sorted(std::vector<int> v) {
    std::sort(v.begin(), v.end()); v.erase(std::unique(v.begin(), v.end()), v.end()); return v;
}

// Rectangular shortest-augmenting-path assignment, following SciPy's tie rules.
// SciPy's implementation: Copyright (c) SciPy Developers. All rights reserved.
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its contributors
//    may be used to endorse or promote products derived from this software
//    without specific prior written permission.
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE, ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
// OR CONSEQUENTIAL DAMAGES (INCLUDING PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
// ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN
// IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
inline std::vector<std::pair<int,int>> assignment(const Matrix &cost) {
    int rows = (int)cost.size(), cols = rows ? (int)cost[0].size() : 0;
    if (!rows || !cols) return {};
    bool transposed = rows > cols;
    int n = transposed ? cols : rows, m = transposed ? rows : cols;
    Vec u(n),v(m),distance(m);std::vector<int> path(m),col(n,-1),row(m,-1),remaining(m);
    std::vector<bool> seen_row(n),seen_col(m);
    for(int start=0;start<n;++start) {
        for(int j=0;j<m;++j)remaining[j]=m-j-1;
        std::fill(seen_row.begin(),seen_row.end(),false);std::fill(seen_col.begin(),seen_col.end(),false);
        std::fill(distance.begin(),distance.end(),std::numeric_limits<double>::infinity());
        int left=m,i=start,sink=-1;double value=0;
        while(sink<0) {
            int selected=-1;double lowest=std::numeric_limits<double>::infinity();seen_row[i]=true;
            for(int k=0;k<left;++k) {
                int j=remaining[k];double candidate=value+(transposed?cost[j][i]:cost[i][j])-u[i]-v[j];
                if(candidate<distance[j]){distance[j]=candidate;path[j]=i;}
                if(distance[j]<lowest || (distance[j]==lowest && row[j]<0)){lowest=distance[j];selected=k;}
            }
            if(selected<0 || !std::isfinite(lowest))throw std::invalid_argument("infeasible assignment");
            value=lowest;int j=remaining[selected];if(row[j]<0)sink=j;else i=row[j];
            seen_col[j]=true;remaining[selected]=remaining[--left];
        }
        u[start]+=value;
        for(int k=0;k<n;++k)if(seen_row[k] && k!=start)u[k]+=value-distance[col[k]];
        for(int j=0;j<m;++j)if(seen_col[j])v[j]-=value-distance[j];
        int j=sink;do {i=path[j];row[j]=i;std::swap(col[i],j);}while(i!=start);
    }
    std::vector<std::pair<int,int>> out;
    for(int i=0;i<n;++i)out.emplace_back(transposed?col[i]:i,transposed?i:col[i]);
    std::sort(out.begin(), out.end()); return out;
}

// The leaf traversal order matters: SciPy's fcluster labels are not sorted by note index.
inline std::vector<int> cluster(const Matrix &e, const Matrix &p, double tau, double lambda) {
    int n=(int)e.size(); if (!n) return {}; if (n==1) return {0};
    Matrix d(2*n-1, Vec(2*n-1, std::numeric_limits<double>::infinity()));
    std::vector<Vec> pn; for (const auto &row:p) pn.push_back(unit(row));
    for (int i=0; i<n; ++i) for (int j=i+1; j<n; ++j) {
        float de=0, dp=0;
        for (size_t k=0;k<e[i].size();++k) de+=(float)e[i][k]*(float)e[j][k];
        for (size_t k=0;k<pn[i].size();++k) dp+=(float)pn[i][k]*(float)pn[j][k];
        float dist=1.0f-de; if (lambda>0) dist+=(float)lambda*(1.0f-dp);
        d[i][j]=d[j][i]=std::max(0.0, (double)dist);
    }
    using Edge = std::pair<double,std::pair<int,int>>;
    std::priority_queue<Edge,std::vector<Edge>,std::greater<Edge>> heap;
    for(int i=0;i<n;++i)for(int j=i+1;j<n;++j)heap.push({d[i][j],{i,j}});
    std::vector<int> size(2*n-1,1), left(2*n-1,-1), right(2*n-1,-1); std::vector<bool> alive(2*n-1,false);
    for(int i=0;i<n;++i) alive[i]=true;
    for (int k=n;k<2*n-1;++k) {
        while(!heap.empty() && (!alive[heap.top().second.first] || !alive[heap.top().second.second]))heap.pop();
        if(heap.empty())throw std::runtime_error("invalid linkage distances");
        auto edge=heap.top();heap.pop();int a=edge.second.first,b=edge.second.second;double best=edge.first;
        left[k]=a; right[k]=b; size[k]=size[a]+size[b];
        for(int i=0;i<k;++i) if(alive[i] && i!=a && i!=b) {
            d[k][i]=d[i][k]=(size[a]*d[a][i]+size[b]*d[b][i])/size[k];heap.push({d[k][i],{i,k}});
        }
        d[k][k]=best; alive[a]=alive[b]=false; alive[k]=true;
    }
    std::vector<int> labels(n,-1); int label=0;
    // Traverse the dendrogram, emitting one label for each subtree below the cut.
    std::vector<std::pair<int,int>> stack{{2*n-2,-1}};
    while(!stack.empty()) {
        auto item=stack.back(); stack.pop_back(); int node=item.first, lab=item.second;
        if(lab<0 && (node<n || d[node][node]<=tau)) lab=label++;
        if(node<n) labels[node]=lab;
        else {stack.emplace_back(right[node],lab); stack.emplace_back(left[node],lab);}
    }
    return labels;
}

inline std::vector<int> chain(const Prediction &p, const std::vector<Note> &notes, const DecodeOptions &opt) {
    int n=(int)notes.size(), next=0; std::vector<int> track(n,-1),owner(n,-1); Vec best(n,-1);
    std::vector<int> order(p.windows.size()); std::iota(order.begin(),order.end(),0);
    std::stable_sort(order.begin(),order.end(),[&](int a,int b){return p.windows[a].start<p.windows[b].start;});
    for(int wi:order) {
        const auto &w=p.windows[wi]; if(w.indices.empty()) continue;
        auto lab=cluster(w.embedding,w.probability,opt.tau,opt.lambda); int count=*std::max_element(lab.begin(),lab.end())+1;
        std::vector<int> gids; for(int id:w.indices) if(owner[id]>=0) gids.push_back(owner[id]); gids=unique_sorted(gids);
        Matrix overlap(count, Vec(gids.size()));
        for(size_t i=0;i<w.indices.size();++i) if(owner[w.indices[i]]>=0) {
            int g=(int)(std::lower_bound(gids.begin(),gids.end(),owner[w.indices[i]])-gids.begin()); overlap[lab[i]][g]++;
        }
        Matrix cost=overlap; for(auto &row:cost) for(double &x:row) x=-x;
        std::vector<int> global(count,-1);
        for(auto rc:assignment(cost)) if(overlap[rc.first][rc.second]>0) global[rc.first]=gids[rc.second];
        for(int &g:global) if(g<0) g=next++;
        for(size_t i=0;i<w.indices.size();++i) {
            int id=w.indices[i], g=global[lab[i]]; owner[id]=g;
            double weight=std::max(.05,std::min(1.0,1-std::abs((notes[id].onset-w.start)/p.window_seconds-.5)*2));
            if(weight>best[id]) {best[id]=weight;track[id]=g;}
        }
    }
    return track;
}

inline std::vector<int> tracklets(const Prediction &p, const std::vector<Note> &notes, const DecodeOptions &opt) {
    auto track=chain(p,notes,opt); std::vector<int> ids;
    for(int t:track) if(t>=0) ids.push_back(t); ids=unique_sorted(ids); int m=(int)ids.size(); if(m<2) return track;
    Matrix e(m,Vec(p.embedding[0].size())), q(m,Vec(p.probability[0].size())); Vec size(m);
    for(size_t i=0;i<track.size();++i) if(track[i]>=0) {
        int t=(int)(std::lower_bound(ids.begin(),ids.end(),track[i])-ids.begin()); ++size[t]; Vec pn=unit(p.probability[i]);
        for(size_t k=0;k<e[t].size();++k)e[t][k]+=p.embedding[i][k];
        for(size_t k=0;k<q[t].size();++k)q[t][k]+=pn[k];
    }
    for(int t=0;t<m;++t) {for(double &x:e[t])x/=size[t];for(double &x:q[t])x/=size[t];}
    std::vector<int> lab(m),rank(m); std::iota(lab.begin(),lab.end(),0); std::iota(rank.begin(),rank.end(),0);
    std::stable_sort(rank.begin(),rank.end(),[&](int a,int b){return size[a]>size[b];});
    std::vector<bool> alive(m,true); Matrix distances(m,Vec(m,std::numeric_limits<double>::infinity()));
    for(int i=0;i<m;++i)for(int j=0;j<m;++j)if(i!=j)distances[i][j]=1-dot(e[i],e[j])+(opt.lambda>0?opt.lambda*(1-dot(q[i],q[j])):0);
    for(int i=600;i<m;++i)for(int j=0;j<m;++j)distances[rank[i]][j]=distances[j][rank[i]]=std::numeric_limits<double>::infinity();
    while(true) {
        int a=-1,b=-1; double best=std::numeric_limits<double>::infinity();
        for(int i=0;i<m;++i)for(int j=0;j<m;++j)if(distances[i][j]<best){best=distances[i][j];a=i;b=j;}
        if(a<0 || best>opt.tau2)break;
        for(size_t k=0;k<e[a].size();++k)e[a][k]=(size[a]*e[a][k]+size[b]*e[b][k])/(size[a]+size[b]);
        for(size_t k=0;k<q[a].size();++k)q[a][k]=(size[a]*q[a][k]+size[b]*q[b][k])/(size[a]+size[b]);
        size[a]+=size[b];alive[b]=false;for(int &x:lab)if(x==b)x=a;
        for(int i=0;i<m;++i)distances[b][i]=distances[i][b]=std::numeric_limits<double>::infinity();
        for(int i=0;i<m;++i)if(alive[i] && i!=a)distances[a][i]=distances[i][a]=1-dot(e[i],e[a])+(opt.lambda>0?opt.lambda*(1-dot(q[i],q[a])):0);
        distances[a][a]=std::numeric_limits<double>::infinity();
    }
    for(int &t:track)if(t>=0)t=lab[std::lower_bound(ids.begin(),ids.end(),t)-ids.begin()]; return track;
}
} // namespace dettato

namespace dettato {
struct Instrument { int program; unsigned families; };
inline const Instrument instruments[73] = {
    {0, 8}, // piano
    {4, 8}, // electric_piano
    {6, 8}, // harpsichord
    {7, 8}, // clavinet
    {46, 136}, // harp
    {8, 24}, // celesta
    {9, 16}, // glockenspiel
    {10, 16}, // music_box
    {11, 16}, // vibraphone
    {12, 16}, // marimba
    {13, 16}, // xylophone
    {14, 16}, // tubular_bells
    {15, 144}, // dulcimer
    {108, 16}, // kalimba
    {112, 16}, // tinkle_bell
    {114, 16}, // steel_drums
    {16, 32}, // organ
    {19, 32}, // church_organ
    {20, 32}, // reed_organ
    {21, 32}, // accordion
    {22, 288}, // harmonica
    {24, 128}, // acoustic_guitar
    {26, 128}, // electric_guitar
    {104, 128}, // sitar
    {105, 128}, // banjo
    {106, 128}, // shamisen
    {107, 128}, // koto
    {32, 1025}, // double_bass
    {33, 1}, // electric_bass
    {38, 2049}, // synth_bass
    {40, 1024}, // violin
    {41, 1024}, // viola
    {42, 1024}, // cello
    {48, 1024}, // string_ensemble
    {50, 3072}, // synth_strings
    {52, 4096}, // voice
    {54, 6144}, // synth_voice
    {56, 2}, // trumpet
    {57, 2}, // trombone
    {58, 2}, // tuba
    {60, 2}, // french_horn
    {61, 2}, // brass_section
    {62, 2050}, // synth_brass
    {64, 256}, // soprano_sax
    {65, 256}, // alto_sax
    {66, 256}, // tenor_sax
    {67, 256}, // baritone_sax
    {68, 256}, // oboe
    {69, 256}, // english_horn
    {70, 256}, // bassoon
    {71, 256}, // clarinet
    {109, 256}, // bagpipe
    {111, 256}, // shanai
    {72, 4}, // piccolo
    {73, 4}, // flute
    {74, 4}, // recorder
    {75, 4}, // pan_flute
    {76, 4}, // blown_bottle
    {77, 4}, // shakuhachi
    {78, 4}, // whistle
    {79, 4}, // ocarina
    {55, 2048}, // orchestra_hit
    {80, 2048}, // synth_lead
    {88, 2048}, // synth_pad
    {96, 2048}, // synth_fx
    {47, 64}, // timpani
    {113, 64}, // agogo
    {115, 64}, // woodblock
    {116, 64}, // taiko
    {117, 64}, // melodic_tom
    {118, 2112}, // synth_drum
    {119, 64}, // reverse_cymbal
    {120, 512}, // sound_effect
};

inline std::vector<Group> decode(const Prediction &p, const std::vector<Note> &notes,
                                 const std::vector<Layer> &head, const DecodeOptions &opt = {}) {
    if(notes.empty()) return {};
    if(p.embedding.size()!=notes.size() || p.probability.size()!=notes.size()) throw std::invalid_argument("prediction size");
    auto tr=tracklets(p,notes,opt); std::vector<int> idx,tids;
    for(size_t i=0;i<tr.size();++i)if(tr[i]>=0){idx.push_back((int)i);tids.push_back(tr[i]);}
    if(idx.empty())return {};
    tids=unique_sorted(tids); int T=(int)tids.size(), N=(int)idx.size(); std::vector<int> inv(N),size(T);
    for(int i=0;i<N;++i){inv[i]=(int)(std::lower_bound(tids.begin(),tids.end(),tr[idx[i]])-tids.begin());size[inv[i]]++;}
    int E=(int)p.embedding[0].size(), K=(int)p.probability[0].size();
    if(K!=73)throw std::invalid_argument("expected 73 instruments");
    for(int pass=0;pass<opt.refine;++pass) {
        std::vector<int> keep;for(int t=0;t<T;++t)if(size[t]>=opt.min_notes)keep.push_back(t);if(keep.size()<2)break;
        Matrix cen(T,Vec(E)), mass(T,Vec(K));
        for(int i=0;i<N;++i) {
            for(int k=0;k<E;++k)cen[inv[i]][k]+=p.embedding[idx[i]][k];
            for(int k=0;k<K;++k)mass[inv[i]][k]+=p.probability[idx[i]][k];
        }
        for(int t:keep){cen[t]=unit(cen[t]);mass[t]=unit(mass[t]);}
        std::fill(size.begin(),size.end(),0);
        for(int i=0;i<N;++i) {
            Vec pn=unit(p.probability[idx[i]]);int best=keep[0];double score=-std::numeric_limits<double>::infinity();
            for(int t:keep){double s=dot(p.embedding[idx[i]],cen[t]);if(opt.lambda>0)s+=opt.lambda*dot(pn,mass[t]);if(s>score){score=s;best=t;}}
            inv[i]=best;++size[best];
        }
    }
    Matrix mass(T,Vec(K));for(int i=0;i<N;++i)for(int k=0;k<K;++k)mass[inv[i]][k]+=p.probability[idx[i]][k];
    std::vector<int> big;for(int t=0;t<T;++t)if(size[t]>=opt.min_notes)big.push_back(t);
    if(big.empty())big.push_back((int)(std::max_element(size.begin(),size.end())-size.begin()));
    Matrix normalized=mass;for(auto &v:normalized)v=unit(v);
    for(int t=0;t<T;++t)if(size[t]>0 && std::find(big.begin(),big.end(),t)==big.end()) {
        int best=big[0];double score=-std::numeric_limits<double>::infinity();
        for(int b:big){double s=dot(normalized[b],normalized[t]);if(s>score){score=s;best=b;}}
        for(int &v:inv)if(v==t)v=best;
    }
    if(opt.absorb>0 && opt.absorb_fraction>0) {
        Matrix emb;for(int i:idx)emb.push_back(unit(p.embedding[i]));
        for(;;) {
            auto active=unique_sorted(inv);int M=(int)active.size();if(M<2)break;
            Matrix cen(M,Vec(E));Vec count(M);
            for(int i=0;i<N;++i){int t=(int)(std::lower_bound(active.begin(),active.end(),inv[i])-active.begin());count[t]++;for(int k=0;k<E;++k)cen[t][k]+=emb[i][k];}
            for(int t=0;t<M;++t)for(double &x:cen[t])x/=count[t];
            std::vector<int> order(M);std::iota(order.begin(),order.end(),0);
            std::stable_sort(order.begin(),order.end(),[&](int a,int b){return count[a]<count[b];}); bool changed=false;
            for(int t:order) {
                double share=count[t]/N;if(share>=opt.absorb_fraction)break;
                std::vector<int> larger;for(int b=0;b<M;++b)if(count[b]>count[t])larger.push_back(b);if(larger.empty())continue;
                bool absorb=false;for(int b:larger)if(1-dot(cen[b],cen[t])<opt.absorb*(1-share/opt.absorb_fraction)){absorb=true;break;}
                if(!absorb)continue;
                for(int b:larger)cen[b]=unit(cen[b]);
                for(int i=0;i<N;++i)if(inv[i]==active[t]) {
                    int best=larger[0];double score=-std::numeric_limits<double>::infinity();
                    for(int b:larger){double s=dot(emb[i],cen[b]);if(s>score){score=s;best=b;}}inv[i]=active[best];
                }
                changed=true;break;
            }
            if(!changed)break;
        }
    }
    std::vector<Group> out;
    for(int t:unique_sorted(inv)) {
        std::vector<int> members;for(int i=0;i<N;++i)if(inv[i]==t)members.push_back(idx[i]);
        Vec pn(K);for(int id:members)for(int k=0;k<K;++k)pn[k]+=p.probability[id][k];
        double sum=std::accumulate(pn.begin(),pn.end(),0.0);Vec z(K);for(int k=0;k<K;++k)z[k]=std::log(std::max(pn[k]/sum,1e-9));
        if(!head.empty() && opt.alpha>0 && !p.hidden.empty()) {
            Vec x(p.hidden[0].size());for(int id:members)for(size_t k=0;k<x.size();++k)x[k]+=p.hidden[id][k]/members.size();
            double mean=0,lo=std::numeric_limits<double>::infinity(),hi=-lo,poly=0;
            for(int id:members){double v=notes[id].pitch/64-1;mean+=v/members.size();lo=std::min(lo,v);hi=std::max(hi,v);}
            double var=0;for(int id:members){double v=notes[id].pitch/64-1;var+=(v-mean)*(v-mean)/members.size();}
            if(members.size()<=3000)for(int a:members){int overlap=0;for(int b:members)if(std::min(std::max(notes[a].offset,notes[a].onset+.02),std::max(notes[b].offset,notes[b].onset+.02))>std::max(notes[a].onset,notes[b].onset)+.03)overlap++;if(overlap>1)poly+=1.0/members.size();}
            x.insert(x.end(),{mean,std::sqrt(var),lo,hi,std::log1p((double)members.size())/5,poly});
            for(size_t l=0;l<head.size();++l) {
                Vec y=head[l].bias;if(y.size()!=head[l].weights.size())throw std::invalid_argument("head dimensions");
                for(size_t j=0;j<y.size();++j){if(head[l].weights[j].size()!=x.size())throw std::invalid_argument("head dimensions");y[j]+=dot(head[l].weights[j],x);if(l+1<head.size())y[j]=.5*y[j]*(1+std::erf(y[j]/std::sqrt(2.0)));}x=std::move(y);
            }
            if(x.size()!=73)throw std::invalid_argument("head output");
            double max=*std::max_element(x.begin(),x.end()),total=0;for(double v:x)total+=std::exp(v-max);double norm=max+std::log(total);
            for(int k=0;k<K;++k)z[k]=opt.alpha*(x[k]-norm)+(1-opt.alpha)*z[k];
        }
        int winner=0;
        if(opt.expected_credit) {
            double max=*std::max_element(z.begin(),z.end());Vec q(K);for(int k=0;k<K;++k)q[k]=std::exp(z[k]-max);
            double best=-1;for(int a=0;a<K;++a){double score=0;for(int b=0;b<K;++b)score+=q[b]*(a==b?1.0:(instruments[a].families & instruments[b].families ? .5:0.0));if(score>best){best=score;winner=a;}}
        } else winner=(int)(std::max_element(z.begin(),z.end())-z.begin());
        auto found=std::find_if(out.begin(),out.end(),[&](const Group &g){return opt.merge && g.instrument==winner;});
        if(found==out.end())out.push_back({winner,members});else found->notes.insert(found->notes.end(),members.begin(),members.end());
    }
    for(auto &g:out)std::sort(g.notes.begin(),g.notes.end());return out;
}
} // namespace dettato
