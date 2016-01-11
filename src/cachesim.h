/*--------------------------------------------------------------------*/
/*--- Cache simulation                                    cg_sim.c ---*/
/*--------------------------------------------------------------------*/
/* Notes:
  - simulates a write-allocate cache
  - (block --> set) hash function uses simple bit selection
  - handling of references straddling two cache blocks:
      - counts as only one cache access (not two)
      - both blocks hit                  --> one hit
      - one block hits, the other misses --> one miss
      - both blocks miss                 --> one miss (not two)
*/

#include "stdio.h"
#include "stdlib.h"
#include "assert.h"
#include "math.h"
#include <list>
#include <map>
#include <vector>
using namespace std;


#define Int int
#define UInt unsigned int
#define Char char
#define UChar unsigned char
#define UWord unsigned int
#define ULong int64_t
#define Long long long int
#define Addr ULong
#define Bool bool
#define True true
#define False false
#define MISS 0
#define HIT 1
#define D(x) ((double)(x))

typedef struct {
  int cachesize;
  double gflops_sec;
  double gups;
  double actual_bytes_per_update;
  double minimum_bytes_per_update_precise;
  double extra;
} stats_t;



static inline int log2(int n)
{
    assert(n > 0);
    unsigned a = 0;
    unsigned exp = 1;
 
    for (a = 0, exp = 1; n > exp && a < 32; exp *= 2, a++);
 
    return a;
}

struct eqstr{
  bool operator()(const ULong s1, const ULong s2) const {
    return s1 == s2;
  }
};

class Cache {
   Int          cachesize;                   /* bytes */
   Int          assoc;
   Int          line_size;              /* bytes */
   Int          sets;
   Int          sets_min_1;
   Int          line_size_bits;
   Int          tag_shift;
   bool         poweroftwo;
   Char         desc_line[128];
   Long*        tags;
   ULong misses;
   ULong naccesses;

   bool fully_associative;
   map<ULong, list<ULong>::iterator> set_tags;
   list<ULong> lru_lst;
   int lru_lst_size;
   

public:

   Cache(Int cachesize, Int assoc, Int line_size)
   {
     init(cachesize, assoc, line_size);
   }
   /* By this point, the size/assoc/line_size has been checked. */
   void init(Int cachesize, Int assoc, Int line_size)
   {
      
      Int i;
      misses=0;naccesses=0;
      this->cachesize      = cachesize;
      this->assoc     = assoc;
      this->line_size = line_size;
      fully_associative = (assoc == cachesize/line_size);
      poweroftwo = (cachesize&(cachesize-1))==0 ;
      
      assert((line_size&(line_size-1))==0); // line size is a power of two
      this->line_size_bits = (log2)(this->line_size);
      this->tags=NULL;

      if(poweroftwo == false)
      {
        if(fully_associative == false)
        {
           fprintf(stderr,  "Non-power of 2 cache must be fully-associative for now\n");
           exit(0); 
        }
        this->tag_shift      = this->line_size_bits;
        sprintf(this->desc_line, "%d B, %d B, %d-way associative",
                                 this->cachesize, this->line_size, this->assoc);
        
      }
      else 
      {
 
        this->sets           = (this->cachesize / this->line_size) / this->assoc;
        this->sets_min_1     = this->sets - 1;
        this->tag_shift      = this->line_size_bits + (log2)(this->sets);
   
        if (this->assoc == 1) {
           sprintf(this->desc_line, "%d B, %d B, direct-mapped",
                                      this->cachesize, this->line_size);
        } else {
           if(fully_associative == false)
             sprintf(this->desc_line, "%d B, %d B, %d-way associative",
                                    this->cachesize, this->line_size, this->assoc);
           else
             sprintf(this->desc_line, "%d B, %d B, %d-way fully-associative",
                                      this->cachesize, this->line_size, this->assoc);
        }
   
        this->tags = (Long *)malloc(sizeof(Long) * this->sets * this->assoc);
   
        for (i = 0; i < this->sets * this->assoc; i++)
           this->tags[i] = -1;
      }

      lru_lst_size=0;
      //printf("%s\n", get_desc_line());
   }

   ~Cache() {if(this->tags) free(this->tags);};
   
   char * get_desc_line() {return desc_line;}
   /* This is done as a macro rather than by passing in the cache_t2 as an
    * arg because it slows things down by a small amount (3-5%) due to all
    * that extra indirection. */
   
   /* The cache and associated bits and pieces. */                             
                                                                               
  
                                                                               
   /* This attribute forces GCC to inline this function, even though it's */   
   /* bigger than its usual limit.  Inlining gains around 5--10% speedup. */   
   int load_line(Addr a)
   {                                                                           
      UInt  set1 = ( a         >> line_size_bits) & (sets_min_1);          
      Long tag2;                                                              
      Int i, j;                                                                
      Bool is_miss = False;                                                    

      naccesses++;

      /* First case: cache is not fully associative */
      if (fully_associative == false)
      {
         Long tag  = a >> tag_shift;                                           
         Long *set = &(tags[set1 * assoc]);                                      
                                                                               
         /* This loop is unrolled for just the first case, which is the most */
         /* common.  We can't unroll any further because it would screw up   */
         /* if we have a direct-mapped (1-way) cache.                        */
         if (tag == set[0]) {                                                  
            return HIT;                                                       
         }                                                                     
         /* If the tag is one other than the MRU, move it into the MRU spot  */
         /* and shuffle the rest down.                                       */
         for (i = 1; i < assoc; i++) {                                       
            if (tag == set[i]) {                                               
               for (j = i; j > 0; j--) {                                       
                  set[j] = set[j - 1];                                         
               }                                                               
               set[0] = tag;                                                   
               return HIT;                                                         
            }                                                                  
         }                                                                     
                                                                               
         /* A miss;  install this tag as MRU, shuffle rest down. */            
         for (j = assoc - 1; j > 0; j--) {                                   
            set[j] = set[j - 1];                                               
         }                                                                     
         set[0] = tag;                                                         
         misses++;
         return MISS;                                                               
      } else {// fully associative
        // single large set
        // find item in the set 
        ULong tag  = (ULong) (a >> tag_shift);

        assert(lru_lst_size == set_tags.size() && lru_lst_size <=  assoc);
        map<ULong, list<ULong>::iterator>::iterator set_tags_it=set_tags.find(tag); 
        if(set_tags_it != set_tags.end()) // if cache hit 
        {
           // just update the lru bit by moving the tag to the front of the list
           list<ULong>::iterator lru_it=set_tags_it->second; 
           assert((*lru_it) == set_tags_it->first); // concistency check
           lru_lst.erase(lru_it); lru_lst.push_front(set_tags_it->first);
           set_tags_it->second=lru_lst.begin(); 
           return HIT;
        } else {// not found
         
          if(lru_lst_size <  assoc) // if some room insert new element, don't replace anyone
          {
             lru_lst.push_front(tag); lru_lst_size++;
             set_tags[tag]=lru_lst.begin(); // assures consistency
          }
          else if(lru_lst_size == assoc) { // if no room,must evict least recently used
             // delete the element
             set_tags.erase(lru_lst.back()); lru_lst.pop_back(); // lru back of the list
             // insert the element
             lru_lst.push_front(tag); set_tags[tag]=lru_lst.begin();

          } else 
              assert(0);


          misses++;
          return MISS;
        }
      } 

      assert(0);
      return MISS;                                                                  
   }

   void evict_line(Addr a)
   {

      UInt set1 = ( a         >> line_size_bits) & (sets_min_1);
      Long tag  = a >> tag_shift;
      Long tag2;
      Int i, j;

      if (fully_associative == false)
        assert(0);
      else
      {
        assert(lru_lst_size == set_tags.size() && lru_lst_size <=  assoc);
        map<ULong, list<ULong>::iterator>::iterator set_tags_it=set_tags.find(tag);
        if(set_tags_it != set_tags.end()) // if cache hit
        {
           // just update the lru bit by moving the tag to the front of the list
           list<ULong>::iterator lru_it=set_tags_it->second;
           assert((*lru_it) == set_tags_it->first); // concistency check
           set_tags.erase(set_tags_it); lru_lst.erase(lru_it); lru_lst_size--; assert(lru_lst_size);
        }
      }
   }

   int load_element(Addr a, Int datasize)
   {
     assert((a%line_size) == 0);
     for(int d=0; d < datasize; d += line_size)
     {
      static ULong prev=0;
      int status=load_line(a+d);
/*
      if(status == MISS) printf("addr:%lld diff:%10lld MISS\n", a+d, a+d-prev);
      else printf("addr:%lld diff:%10lld HIT\n", a+d, a+d-prev);
      prev=a+d;
*/
     }
     return 0;
   }

   int evict_element(Addr a, Int datasize)
   {
     assert((a%line_size) == 0);
     for(int d=0; d < datasize; d += line_size)
       evict_line(a+d);
     return 0;
   }
   
   
   ULong get_misses() {return misses;}
   ULong get_naccesses() {return naccesses;}
   Int get_cachesize() {return cachesize;}

   void resetstat() {naccesses=misses=0;}
   
   void stat(int64_t minnmisses)
   {
     printf("Result statistics\n");
     printf("\taccesses: %lld\n\tallmisses: %lld\n\tredundantmisses: %lld\n", 
            get_naccesses(), 
            get_misses(), get_misses()-minnmisses);
  }
   
};

/*--------------------------------------------------------------------*/
/*--- end                                                 cg_sim.c ---*/
/*--------------------------------------------------------------------*/


