// Run: dettato-grouper-decode-test <capture-dir> <grouper/model.safetensors>
#include "dettato/grouper-decode.h"
#include "yyjson.h"
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>

using namespace dettato;
static std::vector<char> bytes(const std::string &path) {
    std::ifstream f(path,std::ios::binary); if(!f)throw std::runtime_error("cannot open "+path);
    return std::vector<char>(std::istreambuf_iterator<char>(f),{});
}
static yyjson_doc *json_file(const std::string &path) {
    auto b=bytes(path);auto *d=yyjson_read(b.data(),b.size(),0);if(!d)throw std::runtime_error("invalid JSON "+path);return d;
}
static float half(uint16_t h) {
    uint32_t sign=(uint32_t)(h&0x8000)<<16, exp=(h>>10)&31, mant=h&1023, bits;
    if(!exp){if(!mant)bits=sign;else {int e=-14;while(!(mant&1024)){mant<<=1;--e;}mant&=1023;bits=sign|((uint32_t)(e+127)<<23)|(mant<<13);}}
    else if(exp==31)bits=sign|0x7f800000|(mant<<13);
    else bits=sign|((exp+112)<<23)|(mant<<13);
    float out;std::memcpy(&out,&bits,4);return out;
}
template<class T> static std::vector<T> raw(const std::string &path) {
    auto b=bytes(path);if(b.size()%sizeof(T))throw std::runtime_error("bad tensor size "+path);
    std::vector<T> out(b.size()/sizeof(T));std::memcpy(out.data(),b.data(),b.size());return out;
}
static Matrix rows32(const std::vector<float> &v,int width) {
    if(v.size()%width)throw std::runtime_error("bad matrix size");Matrix out(v.size()/width,Vec(width));
    for(size_t i=0;i<v.size();++i)out[i/width][i%width]=v[i];return out;
}
static Matrix rows16(const std::vector<uint16_t> &v,int width) {
    if(v.size()%width)throw std::runtime_error("bad matrix size");Matrix out(v.size()/width,Vec(width));
    for(size_t i=0;i<v.size();++i)out[i/width][i%width]=half(v[i]);return out;
}
static std::vector<Layer> load_head(const std::string &path) {
    auto b=bytes(path);if(b.size()<8)throw std::runtime_error("short safetensors");uint64_t len;std::memcpy(&len,b.data(),8);
    if(len>b.size()-8)throw std::runtime_error("bad safetensors header");
    auto *doc=yyjson_read(b.data()+8,(size_t)len,0);if(!doc)throw std::runtime_error("bad safetensors JSON");auto *root=yyjson_doc_get_root(doc);
    auto tensor=[&](const std::string &name) {
        auto *v=yyjson_obj_get(root,name.c_str());if(!v)throw std::runtime_error("missing "+name);
        auto *dtype=yyjson_obj_get(v,"dtype");if(!dtype || std::string(yyjson_get_str(dtype))!="F32")throw std::runtime_error("head must be F32");
        auto *offset=yyjson_obj_get(v,"data_offsets");size_t lo=(size_t)yyjson_get_uint(yyjson_arr_get(offset,0)),hi=(size_t)yyjson_get_uint(yyjson_arr_get(offset,1));
        if(hi<lo || hi>b.size()-8-len || (hi-lo)%4)throw std::runtime_error("bad head offset");
        std::vector<float> data((hi-lo)/4);std::memcpy(data.data(),b.data()+8+len+lo,hi-lo);return data;
    };
    std::vector<Layer> layers;
    for(int i: {0,2,4}) {
        std::string base="heads.th."+std::to_string(i);auto weights=tensor(base+".weight"),bias=tensor(base+".bias");
        if(bias.empty() || weights.size()%bias.size())throw std::runtime_error("bad head shape");
        Layer l;l.bias.assign(bias.begin(),bias.end());l.weights=rows32(weights,(int)(weights.size()/bias.size()));layers.push_back(std::move(l));
    }
    yyjson_doc_free(doc);return layers;
}
int main(int argc,char **argv) {
    try {
        if(argc!=3){std::cerr<<"usage: dettato-grouper-decode-test <capture> <weights>\n";return 2;}
        if(assignment({{0,-2,-4},{-3,-3,-4}})!=std::vector<std::pair<int,int>>{{0,2},{1,1}})throw std::runtime_error("assignment tie regression");
        std::string dir=argv[1];auto *doc=json_file(dir+"/result.json");auto *root=yyjson_doc_get_root(doc);
        auto *expected=yyjson_obj_get(root,"groups");auto meta=rows32(raw<float>(dir+"/grouper.meta.bin"),12);
        if(meta.empty()) { if(expected && yyjson_arr_size(expected)!=0)throw std::runtime_error("unexpected groups for empty input"); yyjson_doc_free(doc);std::cout<<"PASS "<<dir<<" 0 notes\n";return 0; }
        Prediction p;p.window_seconds=yyjson_get_num(yyjson_obj_get(root,"grouper_window_s"));
        p.embedding=rows32(raw<float>(dir+"/grouper.emb.bin"),32);p.probability=rows32(raw<float>(dir+"/grouper.probs.bin"),73);
        p.hidden=rows16(raw<uint16_t>(dir+"/grouper.h.bin"),176);
        auto times=raw<float>(dir+"/grouper.win.t0.bin");auto counts=raw<int64_t>(dir+"/grouper.win.count.bin");
        auto ids=raw<int64_t>(dir+"/grouper.win.sel.bin");auto emb=rows16(raw<uint16_t>(dir+"/grouper.win.emb.bin"),32);auto prob=rows16(raw<uint16_t>(dir+"/grouper.win.probs.bin"),73);
        if(times.size()!=counts.size())throw std::runtime_error("window count mismatch");size_t pos=0;
        for(size_t i=0;i<times.size();++i){Window w;w.start=times[i];for(int64_t j=0;j<counts[i];++j,++pos){if(pos>=ids.size() || pos>=emb.size() || pos>=prob.size() || ids[pos]<0 || ids[pos]>=(int64_t)meta.size())throw std::runtime_error("bad window indices");w.indices.push_back((int)ids[pos]);w.embedding.push_back(emb[pos]);w.probability.push_back(prob[pos]);}p.windows.push_back(std::move(w));}
        if(pos!=ids.size() || pos!=emb.size() || pos!=prob.size())throw std::runtime_error("window length mismatch");
        std::vector<Note> notes;for(auto &m:meta)notes.push_back({m[0],m[1],m[2]});
        auto actual=decode(p,notes,load_head(argv[2]));
        std::map<int,std::vector<int>> got,want;for(auto &g:actual)got[g.instrument]=g.notes;
        if(!expected)throw std::runtime_error("capture has no grouping");size_t i,max;yyjson_val *item;
        yyjson_arr_foreach(expected,i,max,item){int instrument=(int)yyjson_get_int(yyjson_arr_get(item,0));auto *arr=yyjson_arr_get(item,1);size_t j,n;yyjson_val *id;yyjson_arr_foreach(arr,j,n,id)want[instrument].push_back((int)yyjson_get_int(id));}
        yyjson_doc_free(doc);if(got!=want){std::cerr<<"FAIL grouping differs from capture\n";for(auto &g:got)std::cerr<<" got "<<g.first<<" "<<g.second.size()<<"\n";for(auto &g:want)std::cerr<<" want "<<g.first<<" "<<g.second.size()<<"\n";return 1;}
        std::cout<<"PASS "<<dir<<" "<<notes.size()<<" notes, "<<actual.size()<<" groups\n";return 0;
    } catch(const std::exception &e){std::cerr<<"FAIL "<<e.what()<<"\n";return 1;}
}
