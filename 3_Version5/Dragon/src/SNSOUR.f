*DECK SNSOUR
      SUBROUTINE SNSOUR(MAX1,IG,IPTRK,KPMACR,NANIS,NREG,NMAT,NUNKNO,
     > NGRP,MATCOD,FLUX,SOURCE)
*
*-----------------------------------------------------------------------
*
*Purpose:
* Compute the source for the solution of SN equations.
*
*Copyright:
* Copyright (C) 2007 Ecole Polytechnique de Montreal
* This library is free software; you can redistribute it and/or
* modify it under the terms of the GNU Lesser General Public
* License as published by the Free Software Foundation; either
* version 2.1 of the License, or (at your option) any later version
*
*Author(s): A. Hebert 
*
*Parameters: input
* MAX1    first dimension of FLUX and SOURCE arrays.
* IG      secondary group.
* IPTRK   pointer to the tracking LCM object.
* KPMACR  pointer to the secondary-group related macrolib information.
* NANIS   maximum cross section Legendre order.
* NREG    number of regions.
* NMAT    number of mixtures.
* NUNKNO  number of unknowns per energy group including spherical
*         harmonic terms, interface currents and fundamental
*         currents.
* NGRP    number of energy groups.
* MATCOD  mixture indices.
* FLUX    fluxes.
*
*Parameters: output
* SOURCE  sources.
*
*-----------------------------------------------------------------------
*
      USE GANLIB
*----
*  SUBROUTINE ARGUMENTS
*----
      TYPE(C_PTR) IPTRK,KPMACR
      INTEGER MAX1,IG,NANIS,NREG,NMAT,NUNKNO,NGRP,MATCOD(NREG)
      REAL FLUX(MAX1,NGRP),SOURCE(MAX1,NGRP)
*----
*  LOCAL VARIABLES
*----
      PARAMETER(NSTATE=40,PI4=12.5663706144)
      INTEGER JPAR(NSTATE),P,P2,ILP
      CHARACTER CAN(0:19)*2
*----
*  ALLOCATABLE ARRAYS
*----
      INTEGER, ALLOCATABLE, DIMENSION(:) :: IJJ,NJJ,IPOS
      REAL, ALLOCATABLE, DIMENSION(:) :: XSCAT
      TYPE(C_PTR) IL_PTR,IM_PTR
      INTEGER, POINTER, DIMENSION(:) :: IL,IM
*----
*  DATA STATEMENTS
*----
      DATA CAN /'00','01','02','03','04','05','06','07','08','09',
     >          '10','11','12','13','14','15','16','17','18','19'/
*----
*  SCRATCH STORAGE ALLOCATION
*----
      ALLOCATE(IJJ(0:NMAT),NJJ(0:NMAT),IPOS(0:NMAT))
      ALLOCATE(XSCAT(0:NMAT*NGRP))
*----
*  RECOVER SNT SPECIFIC PARAMETERS.
*----
      CALL LCMGET(IPTRK,'STATE-VECTOR',JPAR)
      IF(JPAR(1).NE.NREG) CALL XABORT('SNSOUR: INCONSISTENT NREG.')
      IF(JPAR(2).NE.NUNKNO) CALL XABORT('SNSOUR: INCONSISTENT NUNKNO.')
      ITYPE=JPAR(6)
      NSCT=JPAR(7)
      IELEM=JPAR(8)
      ISCAT=JPAR(16)
      CALL LCMGPD(IPTRK,'IL',IL_PTR)
      CALL LCMGPD(IPTRK,'IM',IM_PTR)
      CALL C_F_POINTER(IL_PTR,IL,(/ NSCT /))
      CALL C_F_POINTER(IM_PTR,IM,(/ NSCT /))
*----
*  CONSTRUCT THE SOURCE.
*----
      IJJ(0)=0
      NJJ(0)=0
      IPOS(0)=0
      XSCAT(0)=0.0
      IOF0=0
      DO 90 P=1,NSCT
      ILP = IL(P)
      IF(ILP.GT.MIN(ISCAT-1,NANIS)) GO TO 90
      CALL LCMGET(KPMACR,'NJJS'//CAN(ILP),NJJ(1))
      CALL LCMGET(KPMACR,'IJJS'//CAN(ILP),IJJ(1))
      CALL LCMGET(KPMACR,'IPOS'//CAN(ILP),IPOS(1))
      CALL LCMGET(KPMACR,'SCAT'//CAN(ILP),XSCAT(1))
      IF((ITYPE.EQ.2).OR.(ITYPE.EQ.4)) THEN
*----
*  SLAB OR SPHERICAL 1D CASE.
*----
         DO 20 IR=1,NREG
         IBM=MATCOD(IR)
         IF(IBM.LE.0) GO TO 20
         DO 15 IEL=1,IELEM
         IND=(IR-1)*NSCT*IELEM+IELEM*(P-1)+IEL
         JG=IJJ(IBM)
         DO 10 JND=1,NJJ(IBM)
         IF(JG.NE.IG) THEN
            SOURCE(IND,IG)=SOURCE(IND,IG)+FLUX(IND,JG)*
     1      XSCAT(IPOS(IBM)+JND-1)
         ENDIF
         JG=JG-1
   10    CONTINUE
   15    CONTINUE
   20    CONTINUE
      ELSE IF(ITYPE.EQ.3) THEN
*----
*  CYLINDRICAL 1D CASE.
*----
         DO 50 P2=0,P-1
         IF(MOD((P-1)+P2,2).EQ.1) GO TO 50
         IOF0=IOF0+1
         DO 40 IR=1,NREG
         IBM=MATCOD(IR)
         IF(IBM.LE.0) GO TO 40
         IND=(IR-1)*NSCT+IOF0
         JG=IJJ(IBM)
         DO 30 JND=1,NJJ(IBM)
         IF(JG.NE.IG) THEN
            SOURCE(IND,IG)=SOURCE(IND,IG)+FLUX(IND,JG)*
     1      XSCAT(IPOS(IBM)+JND-1)
         ENDIF
         JG=JG-1
   30    CONTINUE
   40    CONTINUE
   50    CONTINUE
      ELSE IF((ITYPE.EQ.5).OR.(ITYPE.EQ.6).OR.(ITYPE.EQ.8)) THEN
*----
*  2D CASES (CARTESIAN OR R-Z).
*----
         NM=IELEM**2
         DO 70 IR=1,NREG
         IBM=MATCOD(IR)
         IF(IBM.LE.0) GO TO 70
         DO 65 IEL=1,NM
         IND=(IR-1)*NSCT*NM+(P-1)*NM+IEL
         JG=IJJ(IBM)
         DO 60 JND=1,NJJ(IBM)
         IF(JG.NE.IG) THEN
            SOURCE(IND,IG)=SOURCE(IND,IG)+FLUX(IND,JG)*
     1      XSCAT(IPOS(IBM)+JND-1)
         ENDIF
         JG=JG-1
   60    CONTINUE
   65    CONTINUE
   70    CONTINUE
      ELSE IF((ITYPE.EQ.7).OR.(ITYPE.EQ.9)) THEN
*----
* 3D CARTESIAN CASE
*----
         NM=IELEM**3
         DO 110 IR=1,NREG
         IBM=MATCOD(IR)
         IF(IBM.LE.0) GO TO 110
         DO 125 IEL=1,NM
         IND=(IR-1)*NSCT*NM+(P-1)*NM+IEL
         JG=IJJ(IBM)
         DO 120 JND=1,NJJ(IBM)
         IF(JG.NE.IG) THEN
            SOURCE(IND,IG)=SOURCE(IND,IG)+FLUX(IND,JG)*
     1      XSCAT(IPOS(IBM)+JND-1)      
         ENDIF
         JG=JG-1
 120     CONTINUE
 125     CONTINUE
 110     CONTINUE
      ELSE
         CALL XABORT('SNSOUR: TYPE OF DISCRETIZATION NOT IMPLEMENTED.')
      ENDIF
   90 CONTINUE
*----
*  SCRATCH STORAGE DEALLOCATION
*----
      DEALLOCATE(XSCAT)
      DEALLOCATE(IPOS,NJJ,IJJ)
      RETURN
      END
