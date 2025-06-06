/*
 * Copyright (c) 2014 Wireless Communications and Networking Group (WCNG),
 * University of Rochester, Rochester, NY, USA.
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: Cristiano Tapparello <cristiano.tapparello@rochester.edu>
 */

#ifndef ENERGY_HARVESTER_CONTAINER_H
#define ENERGY_HARVESTER_CONTAINER_H

#include "ns3/energy-harvester.h"
#include "ns3/object.h"

#include <stdint.h>
#include <vector>

namespace ns3
{
namespace energy
{

class EnergyHarvester;

/**
 * \ingroup energy
 * \brief Holds a vector of ns3::EnergyHarvester pointers.
 *
 * EnergyHarvesterContainer returns a list of EnergyHarvester pointers
 * installed on a node. Users can use this list to access EnergyHarvester
 * objects to obtain the total energy harvested on a node easily.
 *
 * \see NetDeviceContainer
 *
 */
class EnergyHarvesterContainer : public Object
{
  public:
    /// Const iterator for EnergyHarvester container
    typedef std::vector<Ptr<EnergyHarvester>>::const_iterator Iterator;

  public:
    /**
     * \brief Get the type ID.
     * \return The object TypeId.
     */
    static TypeId GetTypeId();
    /**
     * Creates an empty EnergyHarvesterContainer.
     */
    EnergyHarvesterContainer();
    ~EnergyHarvesterContainer() override;

    /**
     * \param harvester Pointer to an EnergyHarvester.
     *
     * Creates a EnergyHarvesterContainer with exactly one EnergyHarvester
     * previously instantiated.
     */
    EnergyHarvesterContainer(Ptr<EnergyHarvester> harvester);

    /**
     * \param harvesterName Name of EnergyHarvester.
     *
     * Creates an EnergyHarvesterContainer with exactly one EnergyHarvester
     * previously instantiated and assigned a name using the Object name service.
     * This EnergyHarvester is specified by its assigned name.
     */
    EnergyHarvesterContainer(std::string harvesterName);

    /**
     * \param a A EnergyHarvesterContainer.
     * \param b Another EnergyHarvesterContainer.
     *
     * Creates a EnergyHarvesterContainer by concatenating EnergyHarvesterContainer b
     * to EnergyHarvesterContainer a.
     *
     * \note Can be used to concatenate 2 Ptr<EnergyHarvester> directly. C++
     * will be calling EnergyHarvesterContainer constructor with Ptr<EnergyHarvester>
     * first.
     */
    EnergyHarvesterContainer(const EnergyHarvesterContainer& a, const EnergyHarvesterContainer& b);

    /**
     * \brief Get an iterator which refers to the first EnergyHarvester pointer
     * in the container.
     *
     * \returns An iterator which refers to the first EnergyHarvester in container.
     *
     * EnergyHarvesters can be retrieved from the container in two ways. First,
     * directly by an index into the container, and second, using an iterator.
     * This method is used in the iterator method and is typically used in a
     * for-loop to run through the EnergyHarvesters.
     *
     * \code
     *   EnergyHarvesterContainer::Iterator i;
     *   for (i = container.Begin (); i != container.End (); ++i)
     *     {
     *       (*i)->method ();  // some EnergyHarvester method
     *     }
     * \endcode
     */
    Iterator Begin() const;

    /**
     * \brief Get an iterator which refers to the last EnergyHarvester pointer
     * in the container.
     *
     * \returns An iterator which refers to the last EnergyHarvester in container.
     *
     * EnergyHarvesters can be retrieved from the container in two ways. First,
     * directly by an index into the container, and second, using an iterator.
     * This method is used in the iterator method and is typically used in a
     * for-loop to run through the EnergyHarvesters.
     *
     * \code
     *   EnergyHarvesterContainer::Iterator i;
     *   for (i = container.Begin (); i != container.End (); ++i)
     *     {
     *       (*i)->method ();  // some EnergyHarvester method
     *     }
     * \endcode
     */
    Iterator End() const;

    /**
     * \brief Get the number of Ptr<EnergyHarvester> stored in this container.
     *
     * \returns The number of Ptr<EnergyHarvester> stored in this container.
     */
    uint32_t GetN() const;

    /**
     * \brief Get the i-th Ptr<EnergyHarvester> stored in this container.
     *
     * \param i Index of the requested Ptr<EnergyHarvester>.
     * \returns The requested Ptr<EnergyHarvester>.
     */
    Ptr<EnergyHarvester> Get(uint32_t i) const;

    /**
     * \param container Another EnergyHarvesterContainer.
     *
     * Appends the contents of another EnergyHarvesterContainer to the end of
     * this EnergyHarvesterContainer.
     */
    void Add(EnergyHarvesterContainer container);

    /**
     * \brief Append a single Ptr<EnergyHarvester> to the end of this container.
     *
     * \param harvester Pointer to an EnergyHarvester.
     */
    void Add(Ptr<EnergyHarvester> harvester);

    /**
     * \brief Append a single Ptr<EnergyHarvester> referred to by its object
     * name to the end of this container.
     *
     * \param harvesterName Name of EnergyHarvester object.
     */
    void Add(std::string harvesterName);

    /**
     * \brief Removes all elements in the container.
     */
    void Clear();

  private:
    void DoDispose() override;

    /**
     * \brief Calls Object::Initialize () for all EnergySource objects.
     */
    void DoInitialize() override;

  private:
    std::vector<Ptr<EnergyHarvester>> m_harvesters; //!< Harvester container
};

} // namespace energy
} // namespace ns3

#endif /* defined(ENERGY_HARVESTER_CONTAINER_H) */
