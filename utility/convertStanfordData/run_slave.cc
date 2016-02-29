#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
int main(int argc, char ** argv)
{
  FILE * f;
  char name[30];
  char command[50];  
  char * dir = get_current_dir_name();
  //char dir[] = "/cs/vml2/jla291/DepthEstimation/OptimaizeDepthMap/"

  if (argc == 1)
  {
    printf("\nSyntax : run_slave num [maxTime maxMem ncpu]\n\n");
    exit(1);
  }
  char maxTime[15];
  char maxMem[10];
  char ncpu[10];
  char funcCmd[100];
  int nodeNum = atoi(argv[1]);

  if (argc == 2)
    sprintf(maxTime, "00:10:00");
  else
    sprintf(maxTime, "%s", argv[3]);
  if (argc <= 3)
    sprintf(maxMem, "4gb");
  else
    sprintf(maxMem, "%s", argv[4]);
  if (argc <= 4)
    sprintf(ncpu, "1");
  else
    sprintf(ncpu, "%s", argv[5]);

  for (int i = 0; i < nodeNum; i++)
  {
    
    sprintf(name,"%d.pbs",i);
    f = fopen(name, "wt");
    fprintf(f,"#!/bin/sh\n");
    fprintf(f,"#PBS -N genLF%d\n",i);
    fprintf(f,"#PBS -m bea\n");
    fprintf(f,"#PBS -M jla291@sfu.ca\n");
    fprintf(f,"#PBS -l mem=%s\n",maxMem);
    fprintf(f,"#PBS -l ncpus=%s\n",ncpu);
    fprintf(f,"#PBS -W x=\"NACCESSPOLICY:SINGLEJOB\"\n");
    fprintf(f,"#PBS -l arch=x86_64\n");
    fprintf(f,"#PBS -l walltime=%s\n",maxTime);
    fprintf(f,"cd %s\n",dir);
    
    //fprintf(f,"/usr/local-linux/bin/matlab -nojvm -nodesktop -nosplash -singleCompThread -r \"run_slave(%d);\"\n",i+1);

    fprintf(f, "LD_LIBRARY_PATH=/cs/vml1/vml/OpenCV-2.0.0/lib:$LD_LIBRARY_PATH\n");

    fprintf(f, "./generateLF %d", (i+3));
    //fprintf(f, "./OptimizeDepthMap %f", float(6.0));
    fclose(f);
        
    sprintf(command,"!chmod +x %d.pbs",i);
    sprintf(command,"qsub %d.pbs",i);
    system(command);
  }
  
  sprintf(command,"cd ..");
  system(command);
  return 0;
}
