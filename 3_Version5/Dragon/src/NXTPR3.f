*DECK NXTPR3
      SUBROUTINE NXTPR3(IPTRK)
*
*-----------------------------------------------------------------------
*
*Purpose:
* To analyse a 3D prismatic geometry from a general 
* 3D geometry analysis.
*
*Copyright:
* Copyright (C) 2006 Ecole Polytechnique de Montreal
* This library is free software; you can redistribute it and/or
* modify it under the terms of the GNU Lesser General Public
* License as published by the Free Software Foundation; either
* version 2.1 of the License, or (at your option) any later version.
*
*Author(s): R. Le Tellier
*
*Parameters: input
* IPTRK   pointer to the nxt tracking (L_TRACK).
*
*-----------------------------------------------------------------------
*
      USE GANLIB
      IMPLICIT NONE
*----
*  SUBROUTINE ARGUMENTS
*----
      TYPE(C_PTR) IPTRK
*----
*  LOCAL VARIABLES
*----
      INTEGER NSTATE,IOUT
      PARAMETER(NSTATE=40,IOUT=6)
      INTEGER GSTATE(NSTATE),ESTATE(NSTATE),KSIGN(3),ICODE(6),NCODE(6),
     1 KTYPE(3)
      INTEGER IZ,IX,IY,NFREG,NFSUR,NDIM,IDIRG,NBOCEL,NBUCEL,IDIAG,
     1 ISAXIS(3),NOCELL(3),NUCELL(3),MAXMSH,MAXMDH,MAXREG,NBTCLS,
     2 MAXPIN,MAXMSP,MAXRSP,MXGSUR,MXGREG,NUNK,IDIR,JJ,NUCELZ,NZP,
     3 N2REG,N2SUR,N2CEL,N2PIN,I,K,NUNK2,NFSURO,ISUR,NUNKO,ITEMP,
     4 ILON,ITYLCM
      REAL RSTATT(NSTATE),ALBEDO(6)
      DOUBLE PRECISION DZ1,DZ2
      CHARACTER NAMASG*9,NAMREC*12
      LOGICAL HALFS(2),SSYM(2),INVER
      CHARACTER CDIR(4)*1
      DATA CDIR /'X','Y','Z','R'/
      TYPE(C_PTR) JPTRK
*----
*  Allocatable arrays
*----
      INTEGER, ALLOCATABLE, DIMENSION(:) :: MATALB,IUNFLD,MATALB2,
     1 IND2T3,KEYMRG,IND2T3F,MATALBF
      REAL, ALLOCATABLE, DIMENSION(:) :: SURVOL1,SURVOLF,ZCORF
      DOUBLE PRECISION, ALLOCATABLE, DIMENSION(:) :: SURVOL,ZCOR,
     1 SURVOL2,SM
*----
* RECOVER STATE-VECTOR
*----
      GSTATE(:NSTATE)=0
      CALL LCMGET(IPTRK,'STATE-VECTOR',GSTATE)
      NFREG=GSTATE(1)
      NFSUR=GSTATE(5)
      IZ=GSTATE(39)
*---
      IF (IZ.EQ.3) THEN
         IX=1
         IY=2
      ELSEIF (IZ.EQ.2) THEN
         IX=3
         IY=1
         CALL XABORT('NXTPR3: ONLY PRIZ IMPLEMENTED WITH NXT.')
      ELSEIF (IZ.EQ.1) THEN
         IX=2
         IY=3
         CALL XABORT('NXTPR3: ONLY PRIZ IMPLEMENTED WITH NXT.')
      ELSE
         CALL XABORT('NXTPR3: ILLEGAL PROJECTION AXIS')
      ENDIF
*----
* RECOVER INFORMATION FROM NXT 3D GEOMETRY ANALYSIS
*----
      CALL LCMGET(IPTRK,'EXCELTRACKOP',RSTATT)
      CALL LCMGET(IPTRK,'SIGNATURE',KSIGN)
      CALL LCMGET(IPTRK,'TRACK-TYPE',KTYPE)
      CALL LCMGET(IPTRK,'ICODE',ICODE)
      CALL LCMGET(IPTRK,'NCODE',NCODE)
      CALL LCMGET(IPTRK,'ALBEDO',ALBEDO)
      JPTRK=LCMDID(IPTRK,'PROJECTION')
      CALL LCMSIX(JPTRK,'NXTRecords',1)
      CALL LCMSIX(IPTRK,'NXTRecords',1)
      WRITE(NAMASG,'(A1,I8.8)') 'G',1
      NAMREC=NAMASG//'DIM'
      ESTATE(:NSTATE)=0
      CALL LCMGET(IPTRK,NAMREC,ESTATE)
      NDIM     =ESTATE( 1)
      IF (NDIM.NE.3) 
     1 CALL XABORT('NXTPR3: NON 3D GEOMETRY')
      IDIRG    =ESTATE( 3)
      NBOCEL   =ESTATE( 4)
      NBUCEL   =ESTATE( 5)
      IDIAG    =ESTATE( 6)
      ISAXIS(1)=ESTATE( 7)
      ISAXIS(2)=ESTATE( 8)
      ISAXIS(3)=ESTATE( 9)
      IF (ISAXIS(IZ).NE.0)
     1 CALL XABORT('NXTPR3: Z+- SYMMETRIES NOT YET TREATED')
      NOCELL(1)=ESTATE(10)
      NOCELL(2)=ESTATE(11)
      NOCELL(3)=ESTATE(12)
      NUCELL(1)=ESTATE(13)
      NUCELL(2)=ESTATE(14)
      NUCELL(3)=ESTATE(15)
      MAXMSH   =ESTATE(16)
      MAXREG   =ESTATE(17)
      NBTCLS   =ESTATE(18)
      MAXPIN   =ESTATE(19)
      MAXMSP   =ESTATE(20)
      MAXRSP   =ESTATE(21)
      IF (NFSUR.NE.ESTATE(22))
     1 CALL XABORT('NXTPR3: INCONSISTENT NUMBER OF OUTER SURFACES')
      IF (NFREG.NE.ESTATE(23))
     1 CALL XABORT('NXTPR3: INCONSISTENT NUMBER OF REGIONS')
      MXGSUR   =ESTATE(24)
      MXGREG   =ESTATE(25)
      NUNK=NFSUR+NFREG+1
      MAXMDH=MAX(MAXMSH,MAXMSP,MAXREG)
*     surface-volumes and mixture indexes
      ALLOCATE(MATALB(NUNK),SURVOL(NUNK),SURVOL1(NUNK))
      CALL LCMGET(IPTRK,'MATALB      ',MATALB)
      CALL LCMGET(IPTRK,'SAreaRvolume',SURVOL)
      CALL XDRSDB(NUNK,SURVOL1,SURVOL,1)
*     cell index and orientation for the cells filling the geometry
      ALLOCATE(IUNFLD(2*2*NBUCEL))
      NAMREC=NAMASG//'CUF'
      CALL LCMGET(IPTRK,NAMREC,IUNFLD)
      NUCELZ=NUCELL(IZ)
      ALLOCATE(IND2T3(NUNK*(MAXMDH*NUCELZ+2)),ZCOR(MAXMDH*NUCELZ+2),
     1 MATALB2(NUNK),SURVOL2(NUNK))
*----
* CONSTRUCT 2D GEOMETRY ANALYSIS
*----
*     CONSTRUCT (2D,Z)->3D INDEX AND FILL IN LEVEL 1-2 DESCRIPTION
      CALL NXT3T2(IPTRK,JPTRK,IX,IY,IZ,NFREG,NFSUR,MAXMDH,NUCELL,
     1     NBUCEL,MXGSUR,MXGREG,MAXPIN,MATALB,SURVOL,IUNFLD,NZP,
     2     N2REG,N2SUR,N2CEL,N2PIN,IND2T3,ZCOR,MATALB2,SURVOL2)
*     NXT LEVEL 0 DESCRIPTION
*     record DIM
      ESTATE(1)=2
      ESTATE(4)=ESTATE(4)/NUCELL(3)
      ESTATE(5)=ESTATE(5)/NUCELL(3)
      ESTATE(9)=0
      ESTATE(12)=0
      ESTATE(13)=NUCELL(IX)
      ESTATE(14)=NUCELL(IY)
      ESTATE(15)=0
      ESTATE(16)=N2CEL
      ESTATE(22)=N2SUR
      ESTATE(23)=N2REG
      NAMREC=NAMASG//'DIM'
      CALL LCMPUT(JPTRK,NAMREC,NSTATE,1,ESTATE)
*     record CUF
      NAMREC=NAMASG//'CUF'
      CALL LCMPUT(JPTRK,NAMREC,2*ESTATE(5),1,IUNFLD(2*NBUCEL+1))
*     record SMX,SMY
      ALLOCATE(SM(NUCELL(IX)+1))
        NAMREC=NAMASG//'SM'//CDIR(IX)
        CALL LCMGET(IPTRK,NAMREC,SM)
        NAMREC=NAMASG//'SM'//CDIR(1)
        CALL LCMPUT(JPTRK,NAMREC,(NUCELL(IX)+1),4,SM)
      DEALLOCATE(SM)
      ALLOCATE(SM(NUCELL(IY)+1))
        NAMREC=NAMASG//'SM'//CDIR(IY)
        CALL LCMGET(IPTRK,NAMREC,SM)
        NAMREC=NAMASG//'SM'//CDIR(2)
        CALL LCMPUT(JPTRK,NAMREC,(NUCELL(IY)+1),4,SM)
      DEALLOCATE(SM)
*---
*  ADDITIONAL RECORDS TO MODIFY/ADD IN /PROJECTION/
*---
*     KEYMRG ARRAY
      CALL LCMLEN(IPTRK,'KEYMRG      ',ILON,ITYLCM)
      ALLOCATE(KEYMRG(ILON))
        CALL LCMGET(IPTRK,'KEYMRG      ',KEYMRG)
        CALL LCMPUT(JPTRK,'KEYMRG      ',N2SUR+N2REG+1,1,
     1       KEYMRG(NFSUR-N2SUR+1))
      DEALLOCATE(KEYMRG)
*     NXT SPECIFIC STATE-VECTOR
      CALL LCMSIX(JPTRK,' ',2)
      GSTATE(1)=N2REG
      GSTATE(2)=N2REG
      GSTATE(5)=N2SUR
      GSTATE(8)=1
      GSTATE(9)=0
      GSTATE(10)=0
      GSTATE(13)=1
      CALL LCMPUT(JPTRK,'STATE-VECTOR',NSTATE,1,GSTATE)
      CALL LCMPUT(JPTRK,'EXCELTRACKOP',NSTATE,2,RSTATT)  
      CALL LCMPUT(JPTRK,'SIGNATURE   ',3,3,KSIGN)
      CALL LCMPUT(JPTRK,'TRACK-TYPE  ',3,3,KTYPE)
      CALL LCMPUT(JPTRK,'ICODE       ',6,1,ICODE)
      CALL LCMPUT(JPTRK,'ALBEDO      ',6,2,ALBEDO)
*---
*  TAKE CARE OF SYMMETRIES ALONG THE PROJECTION AXIS 
*  UPDATE RECORDS ACCORDINGLY
*---
*     MAIN STATE-VECTOR: store the number of z-plan in the prismatic geometry
      CALL LCMSIX(IPTRK,' ',2)
      GSTATE(:NSTATE)=0
      CALL LCMGET(IPTRK,'STATE-VECTOR',GSTATE)
      GSTATE(39)=NZP
*     NCODE ARRAY
      DO JJ=1,2
         HALFS(JJ)=(NCODE(2*(IZ-1)+JJ).EQ.5)
         SSYM(JJ)=((NCODE(2*(IZ-1)+JJ).EQ.10).OR.(HALFS(JJ)))
      ENDDO
      INVER=(SSYM(1).AND.(.NOT.SSYM(2)))
      IF (SSYM(1).OR.SSYM(2)) NCODE(2*IZ)=30
      IF (SSYM(1).AND.SSYM(2)) NCODE(2*IZ-1)=30
      IF (HALFS(1).OR.HALFS(2)) 
     1 CALL XABORT('NXTPR3: SYME NOT SUPPORTED IN PRISMATIC, USE SSYM.')
      CALL LCMPUT(IPTRK,'NCODE       ',6,1,NCODE)
*
      IF (SSYM(1)) IND2T3(NFSUR+2:NFSUR+1+N2REG)=0
      IF (SSYM(2)) THEN
         IND2T3((NZP+1)*NUNK+NFSUR+2:(NZP+1)*NUNK+NFSUR+1+N2REG)=0
      ENDIF
      NFSURO=NFSUR
      NUNKO=NUNK
      IF (SSYM(1)) NFSUR=NFSUR-N2REG
      IF (SSYM(2)) NFSUR=NFSUR-N2REG
      NUNK=NFSUR+NFREG+1
*
      NUNK2=N2SUR+N2REG+1
      ALLOCATE(IND2T3F(NUNK2*(NZP+2)),SURVOLF(NUNK),MATALBF(NUNK))
      DO JJ=0,NFREG
         SURVOLF(NFSUR+JJ+1)=SURVOL1(NFSURO+JJ+1)
         MATALBF(NFSUR+JJ+1)=MATALB(NFSURO+JJ+1)
      ENDDO
      JJ=-1
      ISUR=0
      DO 15 K=0,NZP+1
      DO 10 I=0,NUNK2-1
         JJ=JJ+1
         IDIR=IND2T3(K*NUNKO+I+NFSURO-N2SUR+1)
         IF (IDIR.LT.0) THEN
            ISUR=ISUR+1
            IND2T3F(JJ+1)=-ISUR
            SURVOLF(NFSUR-ISUR+1)=0.25*SURVOL1(NFSURO+IDIR+1)
            MATALBF(NFSUR-ISUR+1)=MATALB(NFSURO+IDIR+1)
         ELSE
            IND2T3F(JJ+1)=IDIR
         ENDIF
 10   CONTINUE
 15   CONTINUE
      IF (ISUR.NE.NFSUR) THEN
         write(*,*) ISUR,NFSUR,NFSURO,N2REG
         CALL XABORT('NXTPR3: NFSUR OVERFLOW.')
      ENDIF   
      GSTATE(5)=NFSUR
      ALLOCATE(ZCORF(NZP+1))
      IF (INVER) THEN
         DO K=0,(NZP+1)/2
            DO I=0,NUNK2-1
               ITEMP=IND2T3F(K*NUNK2+I+1)
               IND2T3F(K*NUNK2+I+1)=IND2T3F((NZP+1-K)*NUNK2+I+1)
               IND2T3F((NZP+1-K)*NUNK2+I+1)=ITEMP
            ENDDO
         ENDDO
         ZCORF=0.0
         DO K=1,NZP
            DZ1=ZCOR(NZP-K+1)
            DZ2=ZCOR(NZP-K+2)
            ZCORF(K+1)=ZCORF(K)+REAL(DZ2-DZ1)
         ENDDO
      ELSE
         CALL XDRSDB(NZP+1,ZCORF,ZCOR,1)
      ENDIF
      CALL LCMPUT(JPTRK,'MATALB      ',NUNK,1,MATALBF)
      CALL LCMPUT(JPTRK,'VOLSUR      ',NUNK,2,SURVOLF)
      CALL LCMPUT(JPTRK,'IND2T3      ',NUNK2*(NZP+2),1,IND2T3F)  
      CALL LCMPUT(JPTRK,'ZCOORD      ',(NZP+1),2,ZCORF)   
      CALL LCMPUT(IPTRK,'STATE-VECTOR',NSTATE,1,GSTATE)
      DEALLOCATE(ZCORF,MATALBF,SURVOLF,IND2T3F)
*----
*  DEALLOCATE MEMORY
*----
      DEALLOCATE(SURVOL2,MATALB2,ZCOR,IND2T3,IUNFLD,SURVOL1,SURVOL,
     1 MATALB)
*
      RETURN
      END
