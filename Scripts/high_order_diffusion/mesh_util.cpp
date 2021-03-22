#include "mesh_util.hpp"
#include <algorithm>
#include <map>
#include <set>

namespace mfem
{

void
IdentifyPeriodicMeshVertices(const Mesh & mesh,
                             const std::vector<Vector> & trans_vecs,
                             Array<int> & v2v,
                             int logging)
{
   int sdim = mesh.SpaceDimension();

   double tol = 1.0e-8;
   double dia = -1.0;

   // map<int,map<int,map<int,int> > > c2v;
   std::set<int> v;
   std::set<int>::iterator si, sj, sk;
   std::map<int,int>::iterator mi;
   std::map<int,std::set<int> >::iterator msi;

   Vector coord(NULL, sdim);

   // map<int,vector<double> > bnd_vtx;
   // map<int,vector<double> > shft_bnd_vtx;

   // int d = 5;
   Vector xMax(sdim), xMin(sdim), xDiff(sdim);
   xMax = xMin = xDiff = 0.0;

   for (int be=0; be<mesh.GetNBE(); be++)
   {
      Array<int> dofs;
      mesh.GetBdrElementVertices(be,dofs);

      for (int i=0; i<dofs.Size(); i++)
      {
         v.insert(dofs[i]);

         coord.SetData(const_cast<double*>(mesh.GetVertex(dofs[i])));
         for (int j=0; j<sdim; j++)
         {
            xMax[j] = std::max(xMax[j],coord[j]);
            xMin[j] = std::min(xMin[j],coord[j]);
         }
      }
   }
   add(xMax, -1.0, xMin, xDiff);
   dia = xDiff.Norml2();

   if ( logging > 0 )
   {
      std::cout << "Number of Boundary Vertices:  " << v.size() << std::endl;

      std::cout << "xMin: ";
      xMin.Print(std::cout,sdim);
      std::cout << "xMax: ";
      xMax.Print(std::cout,sdim);
      std::cout << "xDiff: ";
      xDiff.Print(std::cout,sdim);
   }

   if ( logging > 0 )
   {
      for (si=v.begin(); si!=v.end(); si++)
      {
         std::cout << *si << ": ";
         coord.SetData(const_cast<double*>(mesh.GetVertex(*si)));
         coord.Print(std::cout);
      }
   }

   std::map<int,int>        slaves;
   std::map<int,std::set<int> > masters;

   for (si=v.begin(); si!=v.end(); si++) { masters[*si]; }

   Vector at(sdim);
   Vector dx(sdim);

   for (unsigned int i=0; i<trans_vecs.size(); i++)
   {
      int c = 0;
      if ( logging > 0 )
      {
         std::cout << "trans_vecs = ";
         trans_vecs[i].Print(std::cout,sdim);
      }

      for (si=v.begin(); si!=v.end(); si++)
      {
         coord.SetData(const_cast<double*>(mesh.GetVertex(*si)));

         add(coord, trans_vecs[i], at);

         for (sj=v.begin(); sj!=v.end(); sj++)
         {
            coord.SetData(const_cast<double*>(mesh.GetVertex(*sj)));
            add(at, -1.0, coord, dx);

            if ( dx.Norml2() > dia * tol )
            {
               continue;
            }

            int master = *si;
            int slave  = *sj;

            bool mInM = masters.find(master) != masters.end();
            bool sInM = masters.find(slave)  != masters.end();

            if ( mInM && sInM )
            {
               // Both vertices are currently masters
               //   Demote "slave" to be a slave of master
               if ( logging > 0 )
               {
                  std::cout << "Both " << master << " and " << slave
                       << " are masters." << std::endl;
               }
               masters[master].insert(slave);
               slaves[slave] = master;
               for (sk=masters[slave].begin();
                    sk!=masters[slave].end(); sk++)
               {
                  masters[master].insert(*sk);
                  slaves[*sk] = master;
               }
               masters.erase(slave);
            }
            else if ( mInM && !sInM )
            {
               // "master" is already a master and "slave" is already a slave
               // Make "master" and its slaves slaves of "slave"'s master
               if ( logging > 0 )
               {
                  std::cout << master << " is already a master and " << slave
                       << " is already a slave of " << slaves[slave]
                       << "." << std::endl;
               }
               if ( master != slaves[slave] )
               {
                  masters[slaves[slave]].insert(master);
                  slaves[master] = slaves[slave];
                  for (sk=masters[master].begin();
                       sk!=masters[master].end(); sk++)
                  {
                     masters[slaves[slave]].insert(*sk);
                     slaves[*sk] = slaves[slave];
                  }
                  masters.erase(master);
               }
            }
            else if ( !mInM && sInM )
            {
               // "master" is currently a slave and
               // "slave" is currently a master
               // Make "slave" and its slaves slaves of "master"'s master
               if ( logging > 0 )
               {
                  std::cout << master << " is currently a slave of "
                       << slaves[master]<< " and " << slave
                       << " is currently a master." << std::endl;
               }
               if ( slave != slaves[master] )
               {
                  masters[slaves[master]].insert(slave);
                  slaves[slave] = slaves[master];
                  for (sk=masters[slave].begin();
                       sk!=masters[slave].end(); sk++)
                  {
                     masters[slaves[master]].insert(*sk);
                     slaves[*sk] = slaves[master];
                  }
                  masters.erase(slave);
               }
            }
            else
            {
               // Both vertices are currently slaves
               // Make "slave" and its fellow slaves slaves
               // of "master"'s master
               if ( logging > 0 )
               {
                  std::cout << "Both " << master << " and " << slave
                       << " are slaves of " << slaves[master] << " and "
                       << slaves[slave] << " respectively." << std::endl;
               }

               int master_of_master = slaves[master];
               int master_of_slave  = slaves[slave];

               // Move slave and its fellow slaves to master_of_master
               if ( slaves[master] != slaves[slave] )
               {
                  for (sk=masters[master_of_slave].begin();
                       sk!=masters[master_of_slave].end(); sk++)
                  {
                     masters[master_of_master].insert(*sk);
                     slaves[*sk] = master_of_master;
                  }
                  masters.erase(master_of_slave);
                  slaves[master_of_slave] = master_of_master;
               }
            }
            c++;
            break;
         }
      }
      if ( logging > 0 )
      {
         std::cout << "Found " << c << " possible node";
         if ( c != 1 ) { std::cout << "s"; }
         std::cout <<" to project." << std::endl;
      }
   }
   if ( logging > 0 )
   {
      std::cout << "Number of Master Vertices:  " << masters.size() << std::endl;
      std::cout << "Number of Slave Vertices:   " << slaves.size() << std::endl;
      std::cout << "Master to slave mapping:" << std::endl;
      for (msi=masters.begin(); msi!=masters.end(); msi++)
      {
         std::cout << msi->first << " ->";
         for (si=msi->second.begin(); si!=msi->second.end(); si++)
         {
            std::cout << " " << *si;
         }
         std::cout << std::endl;
      }
      std::cout << "Slave to master mapping:" << std::endl;
      for (mi=slaves.begin(); mi!=slaves.end(); mi++)
      {
         std::cout << mi->first << " <- " << mi->second << std::endl;
      }
   }

   v2v.SetSize(mesh.GetNV());

   for (int i=0; i<v2v.Size(); i++)
   {
      v2v[i] = i;
   }

   for (mi=slaves.begin(); mi!=slaves.end(); mi++)
   {
      v2v[mi->first] = mi->second;
   }
}

Mesh *MakePeriodicMesh(Mesh * mesh, const Array<int> & v2v, int logging)
{
   int dim  = mesh->Dimension();

   if ( logging > 0 )
      std::cout << "Euler Number of Initial Mesh:  "
           << ((dim==3)?mesh->EulerNumber():mesh->EulerNumber2D()) << std::endl;

   Mesh *per_mesh = new Mesh(*mesh, true);

   per_mesh->SetCurvature(1, true);

   // renumber elements
   for (int i = 0; i < per_mesh->GetNE(); i++)
   {
      Element *el = per_mesh->GetElement(i);
      int *v = el->GetVertices();
      int nv = el->GetNVertices();
      for (int j = 0; j < nv; j++)
      {
         v[j] = v2v[v[j]];
      }
   }
   // renumber boundary elements
   for (int i = 0; i < per_mesh->GetNBE(); i++)
   {
      Element *el = per_mesh->GetBdrElement(i);
      int *v = el->GetVertices();
      int nv = el->GetNVertices();
      for (int j = 0; j < nv; j++)
      {
         v[j] = v2v[v[j]];
      }
   }

   per_mesh->RemoveUnusedVertices();
   // per_mesh->RemoveInternalBoundaries();

   if ( logging > 0 )
   {
      std::cout << "Euler Number of Final Mesh:    "
           << ((dim==3)?per_mesh->EulerNumber():per_mesh->EulerNumber2D())
           << std::endl;
   }
   return per_mesh;
}

} // namespace mfem
