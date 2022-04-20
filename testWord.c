#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>


#define MAX_STRING 100
#define TAILLE_MAX 1000000

void ReadWord(char *word, FILE *fin) {
  int a = 0, ch;
  while (!feof(fin)) {
    ch = fgetc(fin);
    if (ch == 13) continue;
    if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
      if (a > 0) {
        if (ch == '\n') ungetc(ch, fin);
        break;
      }
      if (ch == '\n') {
        strcpy(word, (char *)"</s>");
        return;
      } else continue;
    }
    word[a] = ch;
    a++;
    if (a >= MAX_STRING - 1) a--;   // Truncate too long words
  }
  word[a] = 0;
}

int NPhrases(FILE *fin)
{
    int n = 0;
    char chaine[TAILLE_MAX] = "";
    fseek(fin, 0, SEEK_SET);
    while (fgets(chaine, TAILLE_MAX, fin) != NULL) // On lit le fichier tant qu'on ne reçoit pas d'erreur (NULL)
    {
       n = n + 1; // On affiche la chaîne qu'on vient de lire
       
    }

    return n ;

}

int MotsParPhrase(FILE *fin){

    char word[MAX_STRING];
    int np = NPhrases(fin);
    int *temp = malloc(sizeof(int)*np);
    int p = 0;
    fseek(fin, 0, SEEK_SET);
    int n = 0;
    char chaine[TAILLE_MAX] = "";
    fseek(fin, 0, SEEK_SET);
   
       while (1) {
            if (feof(fin) ) break;
            ReadWord(word, fin);
            printf("\n%s\n", word);
            temp[n] = p = p + 1 ;
             if (strcmp(word, "</s>") == 0)
             //if (fgetc(fin)=='\n' && p!=0)
            {
                printf("\n------------\n");
                p = 0;
                n = n + 1;
            }
        }
    for (int i = 0; i < np; i++)
    {
        /* code */
        printf("%d \n",temp[i]-1);

    }
    
}




int main()
{
    char word[MAX_STRING];

    FILE *fin = fopen("./text.txt", "rb");
    if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
    int p = 0;
  
    //int m = NPhrases(fin);
    MotsParPhrase(fin);

    fclose(fin);
    return 0;
}

